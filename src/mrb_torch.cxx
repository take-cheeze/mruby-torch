#include <ATen/Functions.h>
#include <ATen/core/ivalue.h>
#include <ATen/core/stack.h>
#include <ATen/core/List.h>
#include <ATen/core/Dict.h>
#include <ATen/core/TensorBody.h>
#include <ATen/core/dispatch/Dispatcher.h>

#include <torch/cuda.h>

#include <mruby.h>
#include <mruby/array.h>
#include <mruby/class.h>
#include <mruby/data.h>
#include <mruby/hash.h>
#include <mruby/string.h>

namespace {

void free_tensor(mrb_state* mrb, void* ptr) {
  delete static_cast<at::Tensor*>(ptr);
}

const mrb_data_type tensor_type = {
  "at::Tensor", free_tensor,
};

at::Tensor& toTensor(mrb_state *mrb, mrb_value v) {
  mrb_assert(DATA_TYPE(v) == &tensor_type);
  return *reinterpret_cast<at::Tensor*>(DATA_PTR(v));
}

c10::IValue toTorch(mrb_state* mrb, const mrb_value& v, const c10::Argument* arg_sch) {
  if (arg_sch && arg_sch->type()->kind() == c10::TypeKind::OptionalType && mrb_nil_p(v)) {
    return c10::IValue();
  }

  c10::TypePtr ex_type;
  if (arg_sch) {
    if (arg_sch->type()->kind() == c10::TypeKind::OptionalType) {
      ex_type = arg_sch->type()->expect<c10::OptionalType>()->getElementType();
    } else {
      ex_type = arg_sch->type();
    }
  }

  if (mrb_fixnum_p(v)) {
    return at::IValue(mrb_fixnum(v));
  } else if (mrb_float_p(v)) {
    return at::IValue(mrb_float(v));
  } else if (mrb_array_p(v)) {
    if (RARRAY_LEN(v) > 0 && mrb_fixnum_p(RARRAY_PTR(v)[0])) {
      std::vector<int64_t> vec(RARRAY_LEN(v));
      for (mrb_int i = 0; i < RARRAY_LEN(v); ++i) {
        mrb_assert(mrb_fixnum_p(RARRAY_PTR(v)[i]));
        vec[i] = mrb_fixnum(RARRAY_PTR(v)[i]);
      }
      return at::IValue(vec);
    } else {
      at::List<at::Tensor> list;
      for (mrb_int i = 0; i < RARRAY_LEN(v); ++i) {
        list.push_back(toTensor(mrb, RARRAY_PTR(v)[i]));
      }
      return at::IValue(list);
    }
  } else if (mrb_string_p(v) || mrb_symbol_p(v)) {
    mrb_value str = v;
    const std::string cstr = mrb_string_p(v)? mrb_string_value_cstr(mrb, &str) : mrb_sym2name(mrb, mrb_symbol(v));
    if (ex_type && arg_sch->name() == "dtype") {
      at::ScalarType dtype =
#define check(t, name) (cstr == #name)? at::k##name :
        AT_FORALL_SCALAR_TYPES(check)
#undef check
          at::ScalarType::Undefined;
      return at::IValue(int(dtype));
    } else if (ex_type && ex_type->kind() == c10::TypeKind::DeviceObjType) {
      return at::IValue(at::Device(cstr));
    }
    return at::IValue(std::string(cstr));
  } else if (mrb_hash_p(v)) {
    c10::Dict<std::string, at::Tensor> dict;
    mrb_value keys = mrb_hash_keys(mrb, v);
    for (mrb_int i = 0; i < RARRAY_LEN(keys); ++i) {
      mrb_value key = RARRAY_PTR(keys)[i];
      dict.insert(mrb_string_value_cstr(mrb, &key), toTensor(mrb, mrb_hash_get(mrb, v, key)));
    }
    return at::IValue(dict);
  } else if (mrb_type(v) == MRB_TT_DATA && DATA_TYPE(v) == &tensor_type) {
    return at::IValue(*static_cast<at::Tensor*>(DATA_PTR(v)));
  }
  mrb_assert(false);
  return at::IValue();
}

mrb_value toMrb(mrb_state* mrb, const c10::IValue& v) {
  mrb_value ret;
  if (v.isTensor()) {
    at::Tensor* ptr = new at::Tensor(v.toTensor());
    struct RData* d = mrb_data_object_alloc(
        mrb,
        mrb_class_get_under(mrb, mrb_module_get(mrb, "Torch"), "Tensor"),
        ptr,
        &tensor_type);
    ret = mrb_obj_value(d);
  } else {
    mrb_assert(false);
  }
  return ret;
}

mrb_value callBoxed(mrb_state* mrb, mrb_sym name, c10::Stack& stack, mrb_int argc, const mrb_value* argv) {
  using namespace std::string_literals;

  mrb_value keywords = mrb_nil_value();
  if (argc >= 1 && mrb_hash_p(argv[argc - 1])) {
    keywords = mrb_obj_dup(mrb, argv[argc - 1]);
    argc -= 1;
  }

  auto& dispatcher = c10::Dispatcher::singleton();
  auto schema = dispatcher.findSchema(at::OperatorName("aten::"s + mrb_sym2name(mrb, name), ""));
  if (!schema && !stack.empty() && stack.front().isTensor()) {
    schema = dispatcher.findSchema(at::OperatorName("aten::"s + mrb_sym2name(mrb, name), "Tensor"));
  }
  if (!schema) {
    mrb_raisef(mrb, E_NOMETHOD_ERROR, "Couldn't find method: %S", mrb_symbol_value(name));
  }

  const auto& args = schema->schema().arguments();

  for (int i = 0; i < argc; ++i) {
    stack.push_back(toTorch(mrb, argv[i], &args[stack.size()]));
  }

  for (size_t i = stack.size(); i < args.size(); ++i) {
    const c10::Argument& arg = args[i];
    if (mrb_test(keywords)) {
      mrb_value key_val = mrb_hash_get(mrb, keywords, mrb_symbol_value(mrb_intern_cstr(mrb, arg.name().c_str())));
      if (mrb_test(key_val)) {
        stack.push_back(toTorch(mrb, key_val, &arg));
        continue;
      }
    }

    // Fallback to default value
    if (!arg.default_value()) {
      mrb_raisef(mrb, E_ARGUMENT_ERROR, "No default value for argument: %S", mrb_symbol_value(mrb_intern_cstr(mrb, arg.name().c_str())));
    }
    stack.push_back(*arg.default_value());
  }
  mrb_assert(stack.size() == args.size());
  dispatcher.callBoxed(*schema, &stack);

  if (stack.size() == 1) {
    return toMrb(mrb, stack.front());
  }

  mrb_value ret = mrb_ary_new_capa(mrb, stack.size());
  for (const c10::IValue& v : stack) {
    mrb_ary_push(mrb, ret, toMrb(mrb, v));
  }
  return ret;
}

mrb_value torch_dispatch(mrb_state* mrb, mrb_value self) {
  mrb_sym name;
  mrb_value *argv;
  mrb_int argc;
  mrb_get_args(mrb, "n*", &name, &argv, &argc);

  c10::Stack stack;
  return callBoxed(mrb, name, stack, argc, argv);
}

mrb_value tensor_dispatch(mrb_state* mrb, mrb_value self) {
  mrb_sym name;
  mrb_value *argv;
  mrb_int argc;
  mrb_get_args(mrb, "n*", &name, &argv, &argc);

  c10::Stack stack;
  stack.push_back(toTensor(mrb, self));
  return callBoxed(mrb, name, stack, argc, argv);
}

mrb_value tensor_to_s(mrb_state* mrb, mrb_value self) {
  const at::Tensor& t = toTensor(mrb, self);
  std::string str = t.toString();
  return mrb_str_new(mrb, str.data(), str.size());
}

mrb_value tensor_inspect(mrb_state* mrb, mrb_value self) {
  const at::Tensor& t = toTensor(mrb, self);
  std::ostringstream oss;
  oss << t;
  std::string str = oss.str();
  return mrb_str_new(mrb, str.data(), str.size());
}

mrb_value tensor_sizes(mrb_state* mrb, mrb_value self) {
  const at::Tensor& t = toTensor(mrb, self);
  mrb_value ret = mrb_ary_new_capa(mrb, t.dim());
  for (const int64_t i : t.sizes()) {
    mrb_ary_push(mrb, ret, mrb_fixnum_value(i));
  }
  return ret;
}

mrb_value tensor_device(mrb_state* mrb, mrb_value self) {
  const at::Tensor& t = toTensor(mrb, self);
  std::string dev_str = t.device().str();
  return mrb_str_new(mrb, dev_str.data(), dev_str.size());
}

mrb_value tensor_dtype(mrb_state* mrb, mrb_value self) {
  const at::Tensor& t = toTensor(mrb, self);
  at::ScalarType dtype = t.scalar_type();
  switch (dtype) {
#define check(t, name) case at::k ## name: return mrb_symbol_value(mrb_intern_lit(mrb, #name));
    AT_FORALL_SCALAR_TYPES(check)
#undef check

  default:
    return mrb_symbol_value(mrb_intern_lit(mrb, "unknown"));
  }
}

mrb_value cuda_available_p(mrb_state*, mrb_value) {
  return mrb_bool_value(torch::cuda::is_available());
}

mrb_value cuda_device_count(mrb_state*, mrb_value) {
  return mrb_fixnum_value(torch::cuda::device_count());
}

}

extern "C" void mrb_mruby_torch_gem_init(mrb_state* mrb) {
  RClass* t = mrb_define_module(mrb, "Torch");
  mrb_define_module_function(mrb, t, "method_missing", torch_dispatch, MRB_ARGS_ANY());

  RClass* tensor = mrb_define_class_under(mrb, t, "Tensor", mrb->object_class);
  MRB_SET_INSTANCE_TT(tensor, MRB_TT_DATA);
  mrb_define_method(mrb, tensor, "method_missing", tensor_dispatch, MRB_ARGS_ANY());
  mrb_define_method(mrb, tensor, "to_s", tensor_to_s, MRB_ARGS_NONE());
  mrb_define_method(mrb, tensor, "inspect", tensor_inspect, MRB_ARGS_NONE());
  mrb_define_method(mrb, tensor, "sizes", tensor_sizes, MRB_ARGS_NONE());
  mrb_define_method(mrb, tensor, "device", tensor_device, MRB_ARGS_NONE());
  mrb_define_method(mrb, tensor, "dtype", tensor_dtype, MRB_ARGS_NONE());

  RClass* cuda = mrb_define_module_under(mrb, t, "CUDA");
  mrb_define_module_function(mrb, cuda, "available?", cuda_available_p, MRB_ARGS_NONE());
  mrb_define_module_function(mrb, cuda, "device_count", cuda_device_count, MRB_ARGS_NONE());
}

extern "C" void mrb_mruby_torch_gem_final(mrb_state *mrb) {
}

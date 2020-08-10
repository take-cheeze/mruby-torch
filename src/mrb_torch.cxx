#include <ATen/Functions.h>
#include <ATen/core/ivalue.h>
#include <ATen/core/stack.h>
#include <ATen/core/List.h>
#include <ATen/core/Dict.h>
#include <ATen/core/TensorBody.h>
#include <ATen/core/dispatch/Dispatcher.h>

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

c10::IValue toTorch(mrb_state* mrb, const mrb_value& v) {
  if (mrb_fixnum_p(v)) {
    return at::IValue(mrb_fixnum(v));
  } else if (mrb_float_p(v)) {
    return at::IValue(mrb_float(v));
  } else if (mrb_array_p(v)) {
    if (RARRAY_LEN(v) > 0 && mrb_fixnum_p(RARRAY_PTR(v)[0])) {
      std::vector<int64_t> vec;
      for (mrb_int i = 0; i < RARRAY_LEN(v); ++i) {
        mrb_assert(mrb_fixnum_p(RARRAY_PTR(v)[i]));
        vec.push_back(mrb_fixnum(RARRAY_PTR(v)[i]));
      }
      return at::IValue(vec);
    } else {
      at::List<at::Tensor> list;
      for (mrb_int i = 0; i < RARRAY_LEN(v); ++i) {
        list.push_back(toTensor(mrb, RARRAY_PTR(v)[i]));
      }
      return at::IValue(list);
    }
  } else if (mrb_string_p(v)) {
    return at::IValue(std::string(RSTRING_PTR(v), RSTRING_LEN(v)));
  } else if (mrb_hash_p(v)) {
    c10::Dict<std::string, at::Tensor> dict;
    mrb_value keys = mrb_hash_keys(mrb, v);
    for (mrb_int i = 0; i < RARRAY_LEN(keys); ++i) {
      mrb_value key = RARRAY_PTR(keys)[i];
      dict.insert(mrb_string_cstr(mrb, key), toTensor(mrb, mrb_hash_get(mrb, v, key)));
    }
    return at::IValue(dict);
  } else if (mrb_data_p(v) && DATA_TYPE(v) == &tensor_type) {
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

  for (int i = 0; i < argc; ++i) {
    stack.push_back(toTorch(mrb, argv[i]));
  }

  auto& dispatcher = c10::Dispatcher::singleton();
  auto schema = dispatcher.findSchema(at::OperatorName("aten::"s + mrb_sym2name(mrb, name), ""));
  mrb_assert(schema);
  const auto& args = schema->schema().arguments();
  for (size_t i = stack.size(); i < args.size(); ++i) {
    mrb_assert(args[i].default_value());
    stack.push_back(*args[i].default_value());
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
  mrb_get_args(mrb, "na", &name, &argv, &argc);

  c10::Stack stack;
  stack.push_back(toTensor(mrb, self));
  return callBoxed(mrb, name, stack, argc, argv);
}

mrb_value tensor_to_s(mrb_state* mrb, mrb_value self) {
  at::Tensor* t = reinterpret_cast<at::Tensor*>(DATA_PTR(self));
  std::string str = t->toString();
  return mrb_str_new(mrb, str.data(), str.size());
}

mrb_value tensor_inspect(mrb_state* mrb, mrb_value self) {
  at::Tensor* t = reinterpret_cast<at::Tensor*>(DATA_PTR(self));
  std::ostringstream oss;
  oss << *t;
  std::string str = oss.str();
  return mrb_str_new(mrb, str.data(), str.size());
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
}

extern "C" void mrb_mruby_torch_gem_final(mrb_state *mrb) {
}

#include <ATen/Functions.h>
#include <ATen/core/ivalue.h>
#include <ATen/core/stack.h>
#include <ATen/core/TensorBody.h>

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

c10::IValue toTorch(mrb_state* mrb, const mrb_value& v) {
  if (mrb_fixnum_p(v)) {
    return at::IValue(mrb_fixnum(v));
  } else if (mrb_float_p(v)) {
    return at::IValue(mrb_float(v));
  } else if (mrb_array_p(v)) {
    at::List<at::IValue> list;
    for (mrb_int i = 0; i < RARRAY_LEN(v); ++i) {
      list.push_back(toTorch(mrb, RARRAY_PTR(v)[i]));
    }
    return at::IValue(list);
  } else if (mrb_string_p(v)) {
    return at::IValue(std::string(RSTRING_PTR(v), RSTRING_LEN(v)));
  } else if (mrb_hash_p(v)) {
    at::Dict<at::IValue, at::IValue> dict;
    mrb_value keys = mrb_hash_keys(mrb, v);
    for (mrb_int i = 0; i < RARRAY_LEN(keys); ++i) {
      dict.insert(toTorch(mrb, RARRAY_PTR(keys)[i]), toTorch(mrb, v));
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
    
  }
  return ret;
}

mrb_value torch_dispatch(mrb_state* mrb, mrb_value self) {
  mrb_sym name;
  mrb_value *argv;
  mrb_int argc;
  mrb_get_args(mrb, "na", &name, &argv, &argc);

  torch::jit::Stack stack;
  for (int i = 0; i < argc; ++i) {
    stack.push_back(toTorch(mrb, argv[i]));
  }

  auto& dispatcher = c10::Dispatcher::singleton();
  auto schema = dispatcher.findSchema(c10::OperatorName(mrb_sym2name(mrb, name), ""));
  mrb_assert(schema);
  dispatcher.callBoxed(*schema, &stack);

  mrb_value ret = mrb_ary_new_capa(mrb, stack.size());
  for (const c10::IValue& v : stack) {
    mrb_ary_push(mrb, ret, toMrb(mrb, v));
  }

  return RARRAY_LEN(ret) == 1 ? RARRAY_PTR(ret)[0] : ret;
}

mrb_value tensor_dispatch(mrb_state* mrb, mrb_value self) {
  mrb_sym name;
  mrb_value *argv;
  mrb_int argc;
  mrb_get_args(mrb, "na", &name, &argv, &argc);

  torch::jit::Stack stack;
  stack.push_back(toTorch(mrb, self));
  for (int i = 0; i < argc; ++i) {
    stack.push_back(toTorch(mrb, argv[i]));
  }

  auto& dispatcher = c10::Dispatcher::singleton();
  auto schema = dispatcher.findSchema(c10::OperatorName(mrb_sym2name(mrb, name), ""));
  mrb_assert(schema);
  dispatcher.callBoxed(*schema, &stack);

  mrb_value ret = mrb_ary_new_capa(mrb, stack.size());
  for (const c10::IValue& v : stack) {
    mrb_ary_push(mrb, ret, toMrb(mrb, v));
  }

  return RARRAY_LEN(ret) == 1 ? RARRAY_PTR(ret)[0] : ret;
}

mrb_value tensor_to_s(mrb_state* mrb, mrb_value self) {
  at::Tensor* t = reinterpret_cast<at::Tensor*>(DATA_PTR(self));
  std::string str = t->toString();
  return mrb_str_new(mrb, str.data(), str.size());
}

}

void mrb_mruby_torch_gem_init(mrb_state* mrb) {
  RClass* t = mrb_define_module(mrb, "Torch");
  mrb_define_module_function(mrb, t, "method_missing", torch_dispatch, MRB_ARGS_ANY());

  RClass* tensor = mrb_define_class_under(mrb, t, "Tensor", mrb->object_class);
  MRB_SET_INSTANCE_TT(tensor, MRB_TT_DATA);
  mrb_define_method(mrb, tensor, "method_missing", tensor_dispatch, MRB_ARGS_ANY());
  mrb_define_method(mrb, tensor, "to_s", tensor_to_s, MRB_ARGS_NONE());
}

void mrb_mruby_torch_gem_final(mrb_state *mrb) {
}

#include <torch/torch.h>

#include <mruby.h>

namespace {

void free_tensor(mrb_state* mrb, void* ptr)

const mrb_data_type tensor_type = {
  "at::Tensor", 
}

at::IValue toTorch(mrb_state* mrb, const mrb_value& v) {
  if (mrb_fixnum_p(v)) {
    return at::IValue(mrb_fixnum(v));
  } else if (mrb_float_p(v)) {
    return at::IValue(mrb_float(v));
  } else if (mrb_array_p(v)) {
    at::List<IValue> list;
    for (size_t i = 0; i < RARRAY_LEN(i); ++i) {
      list.push_back(toTorch(mrb, RARRAY_PTR(v)[i]));
    }
    return at::IValue(list);
  } else if (mrb_string_p(v)) {
    return at::IValue(std::string(RSTRING_PTR(v), RSTRING_LEN(v)));
  } else if (mrb_hash_p(v)) {
    at::Dict<at::IValue, at::IValue> dict;
    mrb_value keys = mrb_hash_keys(mrb, v);
    for (size_t i = 0; i < RARRAY_LEN(keys); ++i) {
      dict[RARRAY_PTR(keys)[i]] = toTorch(mrb, v);
    }
    return at::IValue(dict);
  } else if (mrb_data_p(v) && DATA_TYPE(v) == &tensor_type) {
    return at::IValue(*static_cast<at::Tensor*>(DATA_PTR(v)));
  }
  mrb_assert(false);
  return at::IValue();
}

mrb_value toMrb(mrb_state* mrb, const at::IValue& v) {
  
}

mrb_value torch_dispatch(mrb_state* mrb, mrb_value self) {
  mrb_sym name;
  mrb_value *argv;
  mrb_int argc;
  mrb_get_args(mrb, "na", &name, &argv, &argc);

  return ret;
}

mrb_value tensor_to_s(mrb_state* mrb, mrb_value self) {
  
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

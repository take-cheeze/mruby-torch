MRuby::Build.new do |conf|
  toolchain :gcc
  enable_debug
  enable_test

  conf.cc.flags << '-fsanitize=address,undefined'
  conf.cxx.flags << '-fsanitize=address,undefined'
  conf.linker.flags << '-fsanitize=address,undefined'

  conf.gem "#{MRUBY_ROOT}/.."
end

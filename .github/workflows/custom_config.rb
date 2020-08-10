MRuby::Build.new do |conf|
  toolchain :gcc
  enable_debug
  enable_test

  conf.cc.flags << '-fsanitize=address,leak,undefined'
  conf.cxx.flags << '-fsanitize=address,leak,undefined'
  conf.linker.flags << '-fsanitize=address,leak,undefined'

  conf.gem "#{MRUBY_ROOT}/.."
end

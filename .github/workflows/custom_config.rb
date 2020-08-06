MRuby::Build.new do |conf|
  toolchain :gcc
  enable_debug
  enable_test
  conf.gem "#{MRUBY_ROOT}/.."
end

MRuby::Gem::Specification.new 'mruby-torch' do |spec|
  spec.license = 'MIT'
  spec.authors = 'take-cheeze'
  spec.version = '1.6.0'

  cache_dir = "#{ENV['XDG_CACHE_HOME'] || "#{ENV['HOME']}/.cache"}/mruby-torch"

  is_macos = false
  if `uname -s`.strip == 'Darwin'
    target = 'cpu'
    libtorch_url = "https://download.pytorch.org/libtorch/#{target}/libtorch-macos-#{spec.version}.zip"
    is_macos = true
  else
    target = 'cu102'
    target_suffix = "%2B\#{target}"
    target_suffix = '' if target == 'cu102'
    libtorch_url = "https://download.pytorch.org/libtorch/#{target}/libtorch-cxx11-abi-shared-with-deps-#{spec.version}#{target_suffix}.zip"
  end
  libtorch_zip = "#{cache_dir}/libtorch-#{spec.version}-#{target}.zip"
  torch_dir = "#{cache_dir}/#{spec.version}-#{target}/libtorch"
  torch_header = "#{torch_dir}/include/ATen/Functions.h"

  cxx.include_paths << "#{torch_dir}/include" << "#{torch_dir}/include/torch/csrc/api/include"

  file libtorch_zip => __FILE__ do |t|
    FileUtils.mkdir_p File.dirname libtorch_zip
    sh "wget --continue '#{libtorch_url}' -O #{libtorch_zip}"
    FileUtils.touch t.name
  end

  file torch_header => libtorch_zip do |t|
    FileUtils.mkdir_p torch_dir
    Dir.chdir File.dirname torch_dir do
      sh "unzip -nq #{libtorch_zip}"
    end
    FileUtils.touch t.name
  end

  cxx.flags << '-std=c++14'

  linker.library_paths << "#{torch_dir}/lib"
  linker.libraries += %w[torch_cpu torch c10]
  if is_macos
    linker.flags += %W[-Xlinker -rpath -Xlinker #{torch_dir}/lib]
  else
    linker.flags << "-Wl,-rpath=#{torch_dir}/lib" <<
      '-Wl,--no-as-needed' << '-lc10_cuda' << '-ltorch_cuda' << '-Wl,--as-needed'
  end

  file "#{dir}/src/mrb_torch.cxx" => torch_header

  ENV['LSAN_OPTIONS'] = "suppressions=#{dir}/leak.txt"
end

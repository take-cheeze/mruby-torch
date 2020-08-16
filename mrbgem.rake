MRuby::Gem::Specification.new 'mruby-torch' do |spec|
  spec.license = 'MIT'
  spec.authors = 'take-cheeze'
  spec.version = '1.6.0'

  target = 'cu102'
  target_suffix = "%2B\#{target}"
  target_suffix = '' if target == 'cu102'
  libtorch_url = "https://download.pytorch.org/libtorch/#{target}/libtorch-cxx11-abi-shared-with-deps-#{spec.version}#{target_suffix}.zip"
  libtorch_zip = "#{build_dir}/libtorch-#{spec.version}-#{target}.zip"
  torch_dir = "#{build_dir}/libtorch"
  torch_header = "#{torch_dir}/include/ATen/Functions.h"

  cxx.include_paths << "#{torch_dir}/include" << "#{torch_dir}/include/torch/csrc/api/include"

  file libtorch_zip => __FILE__ do |t|
    FileUtils.mkdir_p File.dirname libtorch_zip
    sh "wget --continue '#{libtorch_url}' -O #{libtorch_zip}"
    FileUtils.touch t.name
  end

  file torch_header => libtorch_zip do |t|
    Dir.chdir "#{build_dir}" do
      sh "unzip -q -o #{libtorch_zip}"
    end
    FileUtils.touch t.name
  end

  linker.library_paths << "#{torch_dir}/lib"
  linker.libraries << 'torch_cpu' << 'torch' << 'c10_cuda' << 'c10'
  linker.flags << "-Wl,-rpath=#{torch_dir}/lib"

  file "#{dir}/src/mrb_torch.cxx" => torch_header
end

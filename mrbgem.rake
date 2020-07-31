MRuby::Gem::Specification.new 'mruby-torch' do |spec|
  spec.license = 'MIT'
  spec.authors = 'take-cheeze'
  spec.version = '1.5.1'

  libtorch_url = "https://download.pytorch.org/libtorch/cpu/libtorch-cxx11-abi-shared-with-deps-#{spec.version}%2Bcpu.zip"
  libtorch_zip = "#{build_dir}/libtorch-#{spec.version}.zip"
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
      sh "unzip -o libtorch-#{spec.version}.zip"
    end
    FileUtils.touch t.name
  end

  file "#{dir}/src/mrb_torch.cxx" => torch_header
end

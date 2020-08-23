assert 'Torch::CUDA.available?' do
  assert_nothing_raised { Torch::CUDA.available? }
end

assert 'Torch::CUDA.device_count' do
  assert_true Torch::CUDA.device_count >= 0
end

if Torch::CUDA.available?
  assert 'CUDA tensor' do
    assert_nothing_raised { Torch.rand([10, 10], device: 'cuda') }
  end
end

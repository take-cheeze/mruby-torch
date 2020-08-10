assert 'Torch.ones' do
  t = Torch.ones([10, 10])
  assert_equal 'CPUFloatType', t.to_s
end

assert 'Torch.rand' do
  t = Torch.rand([10, 10])
  assert_equal 'CPUFloatType', t.to_s
end

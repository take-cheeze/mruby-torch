assert 'Torch dispatch keyword arguments' do
  t = Torch.rand(size: [10, 10])
  assert_equal 'CPUFloatType', t.to_s
end

assert 'Torch.inspect' do
  t = Torch.ones([2, 2])
  assert_equal " 1  1\n 1  1\n[ CPUFloatType{2,2} ]", t.inspect
end

assert 'Tensor#add' do
  t = Torch.ones([2, 2])
  res = t.add Torch.ones([2, 2])
  assert_equal 'CPUFloatType', res.to_s
end

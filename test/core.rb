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

assert 'Tensor#shape' do
  t = Torch.rand(size: [10, 10])
  assert_equal [10, 10], t.shape
end

assert 'device argument' do
  assert_nothing_raised do
    Torch.rand([10, 10], device: 'cpu')
  end
end

assert 'Tensor#device' do
  assert_equal 'cpu', Torch.rand(size: [10, 10]).device
end

assert 'Tensor#dtype' do
  assert_equal :Float, Torch.rand(size: [10, 10]).dtype
  assert_equal :Double, Torch.rand(size: [10, 10], dtype: :Double).dtype
end

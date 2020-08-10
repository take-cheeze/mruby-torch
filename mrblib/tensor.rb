module Torch
  class Tensor
    def + rhs; self.add rhs end
    def - rhs; self.sub rhs end
    def * rhs; self.mul rhs end
    def / rhs; self.div rhs end
    def / rhs; self.div rhs end

    alias shape sizes
  end
end

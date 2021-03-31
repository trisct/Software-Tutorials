from icecream import ic

class A(object):
    cls_name = 'base_class_A'

class B(A):
    cls_name = 'derived_class_B'

class C(B):
    cls_name = 'second_order_derived_class_C'

class D(A):
    cls_name = 'derived_class_D'
   
ic(A.__mro__)
ic(B.__mro__)
ic(C.__mro__)
ic(D.__mro__)

ic(super(B, B).cls_name)
ic(super(B).cls_name)
ic(super(C, C).cls_name)
ic(super(D, D).cls_name)
ic(super(B, D).cls_name)
ic(super(B, C).cls_name)

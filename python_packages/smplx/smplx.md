# SMPL-X Package

This is a python package that includes SMPL, SMPL-X, SMPL-H and MANO models.

[TOC]

## SMPL

### SMPL Pose Parameters

`SMPL` model is defined in `smplx/smplx/body_models.py`. Its `body_pose` is a tensor of shape `[N, 23*3]`. You can reshape it to `[N, 23, 3]` for better understanding. After this reshaping, each index `i` in dim 1 corresponds to a joint, and each index `j` in dim 2 corresponds to an axis w.r.t. which to rotate.

Note that, the rotation of a joint is relative to its parent node.

| Index | Part                        | Parent |
| ----- | --------------------------- | ------ |
| 1     | whole left leg              | root   |
| 2     | whole right leg             | root   |
| 3     | upper body above waist      | root   |
| 4     | lower left leg              | 0      |
| 5     | lower right leg             | 1      |
| 6     | upper body above stomach    | 2      |
| 7     | left foot                   | 3      |
| 8     | right foot                  | 4      |
| 9     | upper body above chest      | 5      |
| 10    | left foot toes              | 6      |
| 11    | right foot toes             | 7      |
| 12    | above neck                  | 8      |
| 13    | left arm from shoulder      | 8      |
| 14    | right arm from shoulder     | 8      |
| 15    | head                        | 11     |
| 16    | left arm from biceps        | 12     |
| 17    | right arm from biceps       | 13     |
| 18    | left arm from elbow         | 15     |
| 19    | right arm from elbow        | 16     |
| 20    | left arm from forearm       | 17     |
| 21    | right arm from forearm      | 18     |
| 22    | left palm, excluding thumb  | 19     |
| 23    | right palm, excluding thumb | 20     |

### SMPL Model Usage

For an example, use

```shell
python examples/my_try.py
```

Some key steps are:

__(1) create a model__ with

```python
model = smplx.create(model_folder, model_type='smpl', gender='neutral')
```

This returns a `SMPL` class instance object, which is a derived class of `torch.nn.Module`.

__(2) Its forward call__ is actually a mapping from SMPL parameters to the final SMPL model output, e.g. vertices. If some parameters are not given, then the ones stored in the model will be used. The parameters it uses are

| Name               | Type           | Shape                 | Function                                                     |
| ------------------ | -------------- | --------------------- | ------------------------------------------------------------ |
| `betas`            | `torch.Tensor` | [N, num_betas] (=10)  | body shape parameters (batched)                              |
| `body_pose`        | `torch.Tensor` | [N, num_joints] (=69) | joint rotation angles (23 joints, 3 directions each) (batched) |
| `global_orient`    | `torch.Tensor` |                       |                                                              |
| `transl`           | `torch.Tensor` |                       |                                                              |
| `return_verts`     | `bool`         |                       |                                                              |
| `return_full_pose` | `bool`         |                       |                                                              |

__(3) Its returned object__ by `forward` is a `SMPLOutput` class instance object. Some of its attributes are listed below.
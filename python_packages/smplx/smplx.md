# SMPL-X Package

This is a python package that includes SMPL, SMPL-X, SMPL-H and MANO models.

### SMPL Usage

`SMPL` model is defined in `smplx/smplx/body_models.py`. Its `body_pose` is a tensor of shape `[N, 23*3]`. You can reshape it to `[N, 23, 3]` for better understanding. After this reshaping, each index `i` in dim 1 corresponds to a joint, and each index `j` in dim 2 corresponds to an axis w.r.t. which to rotate.

Note that, the rotation of a joint is relative to its parent node.

| Index | Part                        | Parent |
| ----- | --------------------------- | ------ |
| 0     | whole left leg              | root   |
| 1     | whole right leg             | root   |
| 2     | upper body above waist      | root   |
| 3     | lower left leg              | 0      |
| 4     | lower right leg             | 1      |
| 5     | upper body above stomach    | 2      |
| 6     | left foot                   | 3      |
| 7     | right foot                  | 4      |
| 8     | upper body above chest      | 5      |
| 9     | left foot toes              | 6      |
| 10    | right foot toes             | 7      |
| 11    | above neck                  | 8      |
| 12    | left arm from shoulder      | 8      |
| 13    | right arm from shoulder     | 8      |
| 14    | head                        | 11     |
| 15    | left arm from biceps        | 12     |
| 16    | right arm from biceps       | 13     |
| 17    | left arm from elbow         | 15     |
| 18    | right arm from elbow        | 16     |
| 19    | left arm from forearm       | 17     |
| 20    | right arm from forearm      | 18     |
| 21    | left palm, excluding thumb  | 19     |
| 22    | right palm, excluding thumb | 20     |


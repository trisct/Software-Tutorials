# GAPS: Voxelization

This follows the instructions from [LDIF](https://github.com/google/ldif).



You can do

```
msh2df input.ply tmp.grd -estimate_sign -spacing 0.002 -v
grd2msh tmp.grd output.ply
rm tmp.grd
```


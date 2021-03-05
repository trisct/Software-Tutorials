#for fname in a.txt b.txt c.txt
#do
#	echo $fname
#done


# listing files, loops and string compare
for fname in *
do
    if [ "$fname" != "2_shell_script.sh" ]
    then
        echo $fname
    fi
done


# names
echo
echo "# arguments called with ---->  ${@}     "
echo "# \$1 ---------------------->  $1       "
echo "# \$2 ---------------------->  $2       "
echo "# path to me --------------->  ${0}     "
echo "# parent path -------------->  ${0%/*}  "
echo "# my name ------------------>  ${0##*/} "
echo

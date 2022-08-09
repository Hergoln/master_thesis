# $1 -> path to executable
# $2 -> input directory path
# $3 -> output directory path
find $2 -type f -exec ./single_lex.sh $1 {} $3 \;
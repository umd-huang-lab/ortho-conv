wget -nc http://cs231n.stanford.edu/tiny-imagenet-200.zip
unzip tiny-imagenet-200.zip
rm -r ./tiny-imagenet-200/test
python3 val_format.py
rm -r ./tiny-imagenet-200/val
find . -name "*.txt" -delete
rm tiny-imagenet-200.zip
mv tiny-imagenet-200 tiny-imagenet
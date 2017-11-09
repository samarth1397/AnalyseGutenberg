python3 Download.py
echo "\n\n\n\n"
echo "Downloading Complete"
echo "Analysis of content will begin shortly........."
python3 Analyse-Content.py
echo "\n\n\n\n"
echo "Analysis completed"

echo "\n\n\n\n"
echo "Analysis of style will begin shortly........."

python3 Analyse-Style.py
echo "\n\n\n\n"
echo "Analysis completed"

echo "Deleting files"
rm -rf author*


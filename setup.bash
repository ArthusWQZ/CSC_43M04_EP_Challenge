conda config --add envs_dirs /Data/arthus.wauquiez/conda/envs
conda create -y -n Modal python=3.11
conda activate Modal
conda install -y -c conda-forge uv

# uv export --format requirements.txt --no-hashes -o requirements.txt
uv pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu128 --index-strategy unsafe-best-match
uv pip install -e .
uv pip install gdown

# Download and extract data
cd /Data/arthus.wauquiez/
gdown 1SlRJBD6cyXMr5772kOKe5xXAU9Scu5vR -O data.zip
unzip data.zip -d _tmp_extract
# If the zip has a single top-level folder, promote it; otherwise wrap contents
items=(_tmp_extract/*)
if [ ${#items[@]} -eq 1 ] && [ -d "${items[0]}" ]; then
    mv "${items[0]}" processed_data
    rmdir _tmp_extract
else
    mv _tmp_extract processed_data
fi
rm -f data.zip

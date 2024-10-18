for file in *.mp4; do
    base_name="${file%.mp4}"  # Get the base name without extension
    mkdir $base_name
    ffmpeg -i "$file" -vf "fps=4" "$base_name/${base_name}_%04d.jpg"
done


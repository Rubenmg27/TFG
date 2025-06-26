input_path = "submission_sucio.csv"
output_path = "submission.csv"

with open(input_path, "r") as f_in, open(output_path, "w") as f_out:
    seen_header = False
    for line in f_in:
        if line.strip() == "image_id,isup_grade":
            if not seen_header:
                f_out.write(line)
                seen_header = True
            # si ya vimos el header, lo ignoramos
        else:
            f_out.write(line)

print(f"Archivo limpio guardado en: {output_path}")

from src.engine import process_data

def generate_recs():
    processed_indices = [0, 1, 2, 3]
    last_date = "2023-01-01"
    updated_indices = process_data(processed_indices, last_date)

    print("Updated processed indices:", updated_indices)


def main():
    generate_recs()

main()
import requests
import sys

def download_pdb(pdb_id, save_path="."):
    """
    Downloads a PDB file from the OPM database for a given PDB ID.
    
    Args:
        pdb_id (str): The PDB ID of the membrane protein.
        save_path (str): Directory to save the downloaded PDB file.
    """
    url = f'https://biomembhub.org/shared/opm-assets/pdb/{pdb_id}.pdb'
    file_path = f"{save_path}/{pdb_id}.pdb"
    
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()  # Raise an error for bad status codes
        
        with open(file_path, 'wb') as file:
            for chunk in response.iter_content(chunk_size=8192):
                file.write(chunk)
        
        print(f"PDB file {pdb_id} downloaded successfully and saved to {file_path}")
    except requests.exceptions.RequestException as e:
        print(f"Error downloading PDB file {pdb_id}: {e}")

def main():
 
    if len(sys.argv) < 2:
        print("Usage: python %s pdb_id1 pdb_id2 ..."%sys.argv[0])
        sys.exit(1)

    for pdb_id in sys.argv[1:]:
        download_pdb(pdb_id)

main()
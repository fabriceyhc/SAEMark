# this file is utils to the main program
import json
import hashlib
from text_generation import Client

def generate_pseudo_number(model_name, seed):
    # use the SHA-256 hash function to generate the pseudo number
    sha_signature = hashlib.sha256((model_name + str(seed)).encode()).digest()

    # covert the hash to a big integer, then mod to some range
    pseudo_random_int = int.from_bytes(sha_signature, byteorder='big') % (2**32)

    # normalize the number to [0, 1]
    pseudo_random_float = pseudo_random_int / (2**32)
    
    return pseudo_random_float

# use the pseudo number to generate a more precise number
def generate_precise_number(model_name, seed, low_range, high_range):
    pseudo_random_float = generate_pseudo_number(model_name, seed)
    if pseudo_random_float < 0.5:
        return low_range[0] + pseudo_random_float * (low_range[1] - low_range[0]) * 2
    else:
        return high_range[0] + (pseudo_random_float - 0.5) * (high_range[1] - high_range[0]) * 2

# use the private key to generate the public key
def get_public_key(private_key):
    # use the SHA-256 algorithm to hash the string
    hash_object = hashlib.sha256(private_key.encode())
    # get the hex way of hash object
    hex_dig = hash_object.hexdigest()
    return hex_dig[-10:]
    
# use the public key to generate low edge and high edge
# the low range and high range are calcualted by standard diviation
# the standard deviation is set from 0.5σ to 2σ
def generate_low_high_edges(private_key, min_val=0.5, max_val=1.5, mu=0.1271, sigma=0.0204):
    # use the SHA-256 algorithm to hash the string
    hash_object = hashlib.sha256(private_key.encode())
    # get the hex way of hash object
    hex_dig = hash_object.hexdigest()
    # change the hex to int
    int_digest = int(hex_dig, 16)
    # change the hash between 0 and 100
    range_0_to_100 = int_digest % 101
    # change the value to min_val and max_val
    scaled_value = ((range_0_to_100 / 100) * (max_val - min_val)) + min_val
    # get the round2 of scaled value
    scaled_value = round(scaled_value, 2)
    # get the min, max edges of the private key
    min_edge = round(mu - sigma * scaled_value, 4)
    max_edge = round(mu + sigma * scaled_value, 4)
    return min_edge, max_edge
    
def read_dataset(file_path, start, end):
    with open(file_path, 'r', encoding='utf-8') as f_dataset:
        dataset_lines = f_dataset.readlines()
        
    # use .strip() to clearify dataset_lines
    dataset_lines = [dataset_line.strip() for dataset_line in dataset_lines][start:end]
    
    result = list()
    for dataset_line in dataset_lines:
        line = json.loads(dataset_line)
        result.append(line)
        
    return result

def output_dataset_in_line(filepath, item):
    with open(filepath, 'a') as f_output:
        line = json.dumps(item, ensure_ascii=False)
        f_output.write(f'{line}\n')

def output_dataset(filepath, content):
    with open(filepath, 'a') as f_output:
        for item in content:
            line = json.dumps(item, ensure_ascii=False)
            f_output.write(f'{line}\n')
            
def generate(prompt, generation_config, ip, port):
    client = Client(base_url=f"http://{ip}:{port}", timeout=60)
    
    response = client.generate(
        prompt=prompt, **generation_config
    )
    response = response.model_dump()
    # return response['generated_text']
    res = list()
    for t in response['details']['best_of_sequences']:
        res.append(t['generated_text'])
    return res

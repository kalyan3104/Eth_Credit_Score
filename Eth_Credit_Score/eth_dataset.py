import os
import random
import time
import requests
import pandas as pd
from tqdm import tqdm
from eth_utils import to_checksum_address
from eth_account import Account
from concurrent.futures import ThreadPoolExecutor, as_completed

# Import your existing credit scorer (make sure this module exists)
try:
    from Eth_Credit_Score import EthereumCreditScorer
except ImportError:
    # Fallback mock scorer if import fails
    class EthereumCreditScorer:
        def __init__(self):
            print("Using mock EthereumCreditScorer - implement your real scorer")
            
        def predict_score(self, wallet_data):
            """Mock scoring function - replace with your actual implementation"""
            base_score = (
                300 + 
                (wallet_data.get('account_age_days', 0) / 365 * 10) + 
                (wallet_data.get('eth_balance', 0) * 5) +
                (wallet_data.get('tx_count', 0) * 0.02) +
                (wallet_data.get('tx_success_rate', 0) * 100) +
                (wallet_data.get('defi_interactions', 0) * 2) +
                (wallet_data.get('token_diversity', 0) * 5) -
                (wallet_data.get('malicious_score', 0) * 200) -
                (wallet_data.get('mixer_score', 0) * 150)
            )
            return max(300, min(850, base_score))

class EthereumCreditDatasetGenerator:
    def __init__(self):
        self.scorer = EthereumCreditScorer()
        self.wallets = []
        self.dataset = []
        self.failed_addresses = []
        self.etherscan_api_key = None
        self.last_request_time = 0
        
    def configure_etherscan(self, api_key):
        """Set Etherscan API key with rate limiting"""
        self.etherscan_api_key = api_key
        self.last_request_time = time.time()
        
    def _rate_limited_request(self):
        """Ensure we don't exceed 5 requests/second"""
        elapsed = time.time() - self.last_request_time
        if elapsed < 0.2:  # 5 requests per second
            time.sleep(0.2 - elapsed)
        self.last_request_time = time.time()
    
    def generate_random_wallets(self, count=10000):
        """Generate random Ethereum wallet addresses"""
        print(f"Generating {count} random wallet addresses...")
        self.wallets = [Account.create() for _ in range(count)]
        return self.wallets
    
    def load_from_csv(self, filepath, sample_size=None):
        """Load real addresses from CSV file"""
        print(f"Loading addresses from {filepath}...")
        try:
            df = pd.read_csv(filepath)
            if 'address' not in df.columns:
                raise ValueError("CSV must contain 'address' column")
                
            addresses = df['address'].tolist()
            if sample_size:
                addresses = random.sample(addresses, min(sample_size, len(addresses)))
                
            self.wallets = [{'address': addr} for addr in addresses]
            print(f"Loaded {len(self.wallets)} addresses from CSV")
            return self.wallets
        except Exception as e:
            print(f"Error loading CSV: {e}")
            raise
    
    def fetch_from_etherscan(self, sample_size=1000):
        """Fetch real addresses from Etherscan API"""
        if not self.etherscan_api_key:
            raise ValueError("Etherscan API key not configured")
            
        print("Fetching real addresses from Etherscan...")
        addresses = set()
        
        # Get latest block number
        latest_block = self._etherscan_request(
            module="proxy",
            action="eth_blockNumber"
        )
        latest_block = int(latest_block['result'], 16)
        
        # Scan recent blocks (last 2000 blocks, every 10th block)
        block_range = range(latest_block - 2000, latest_block, 10)
        for block_num in tqdm(block_range, desc="Scanning blocks"):
            try:
                block_data = self._etherscan_request(
                    module="proxy",
                    action="eth_getBlockByNumber",
                    tag=hex(block_num),
                    boolean="true"
                )
                
                if block_data['result'] and 'transactions' in block_data['result']:
                    for txn in block_data['result']['transactions']:
                        if 'from' in txn:
                            addresses.add(txn['from'])
                        if 'to' in txn and txn['to']:
                            addresses.add(txn['to'])
                
                if len(addresses) >= sample_size:
                    break
                    
            except Exception as e:
                print(f"Error processing block {block_num}: {e}")
                continue
        
        if not addresses:
            raise ValueError("No addresses collected from Etherscan")
            
        self.wallets = [{'address': addr} for addr in random.sample(list(addresses), min(sample_size, len(addresses)))]
        print(f"Collected {len(self.wallets)} unique addresses")
        return self.wallets
    
    def _etherscan_request(self, module, action, **params):
        """Make rate-limited Etherscan API request"""
        self._rate_limited_request()
        
        base_url = "https://api.etherscan.io/api"
        params['apikey'] = self.etherscan_api_key
        
        try:
            response = requests.get(
                base_url,
                params={'module': module, 'action': action, **params},
                timeout=15
            )
            response.raise_for_status()
            data = response.json()
            
            if data.get('status') == '0' and data.get('message') != 'No transactions found':
                raise ValueError(f"Etherscan API error: {data.get('result', 'Unknown error')}")
                
            return data
            
        except requests.exceptions.RequestException as e:
            raise ValueError(f"API request failed: {str(e)}")
    
    def _generate_mock_features(self, address):
        """Generate complete mock wallet features including all required fields"""
        features = {
            'account_age_days': random.choice([
                *[random.uniform(1, 30) for _ in range(20)],
                *[random.uniform(30, 180) for _ in range(30)],
                *[random.uniform(180, 365) for _ in range(25)],
                *[random.uniform(365, 365*3) for _ in range(15)],
                *[random.uniform(365*3, 365*5) for _ in range(10)]
            ]),
            'eth_balance': random.choice([
                *[random.uniform(0, 0.1) for _ in range(40)],
                *[random.uniform(0.1, 1) for _ in range(30)],
                *[random.uniform(1, 10) for _ in range(20)],
                *[random.uniform(10, 100) for _ in range(8)],
                *[random.uniform(100, 1000) for _ in range(2)]
            ]),
            'tx_count': random.choice([
                *[random.randint(0, 10) for _ in range(20)],
                *[random.randint(10, 100) for _ in range(30)],
                *[random.randint(100, 1000) for _ in range(30)],
                *[random.randint(1000, 10000) for _ in range(15)],
                *[random.randint(10000, 100000) for _ in range(5)]
            ]),
            'tx_success_rate': random.uniform(0.85, 1.0),
            'defi_interactions': random.randint(0, 50),
            'token_diversity': random.randint(0, 20),
            'malicious_score': random.uniform(0, 0.1) if random.random() < 0.9 else random.uniform(0.1, 1.0),
            'mixer_score': random.uniform(0, 0.1) if random.random() < 0.95 else random.uniform(0.1, 1.0),
            'delegations_given': random.randint(0, 5) if random.random() < 0.3 else 0,
            'delegations_received': random.randint(0, 3) if random.random() < 0.3 else 0,
            'delegation_success_rate': random.uniform(0.7, 1.0) if random.random() < 0.3 else 0,
            'zkp_usage': 1 if random.random() < 0.2 else 0
        }
        
        # Calculate derived features
        features.update({
            'delegation_capacity': min(features['eth_balance'] * 0.1, 100),
            'staked_collateral': min(features['eth_balance'] * 0.05, 50),
            'reputation_score': random.uniform(0, 1) if features['tx_count'] > 10 else 0
        })
        
        return features
    
    def process_wallet(self, wallet, use_mock_data=True):
        """Process a single wallet to generate credit data"""
        try:
            address = wallet.address if hasattr(wallet, 'address') else wallet['address']
            
            # Get wallet data (always use mock for this example)
            wallet_data = self._generate_mock_features(address)
            
            # Generate score
            score = self.scorer.predict_score(wallet_data)
            
            # Create output record
            record = {
                'address': address,
                'score': score,
                **wallet_data
            }
            
            if hasattr(wallet, 'key'):
                record['private_key'] = wallet.key.hex()
                
            return record
            
        except Exception as e:
            address = wallet.address if hasattr(wallet, 'address') else wallet['address']
            self.failed_addresses.append(address)
            print(f"Error processing {address}: {str(e)}")
            return None
    
    def generate_dataset(self, count=1000, use_mock_data=True, workers=4):
        """Generate the complete dataset"""
        start_time = time.time()
        
        if not self.wallets:
            if use_mock_data:
                self.generate_random_wallets(count)
            else:
                try:
                    self.fetch_from_etherscan(count)
                except Exception as e:
                    print(f"Failed to fetch real addresses: {e}")
                    print("Falling back to mock data...")
                    self.generate_random_wallets(count)
                    use_mock_data = True
        
        print(f"Processing {len(self.wallets)} wallets ({'mock' if use_mock_data else 'real'} data)...")
        
        # Process in parallel
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = [executor.submit(self.process_wallet, wallet, use_mock_data) 
                      for wallet in self.wallets[:count]]
            
            results = []
            for future in tqdm(as_completed(futures), total=len(futures)):
                result = future.result()
                if result:
                    results.append(result)
        
        self.dataset = results
        
        if not self.dataset:
            raise ValueError("No wallets processed successfully")
            
        df = pd.DataFrame(self.dataset)
        
        # Print stats
        duration = time.time() - start_time
        print(f"\nCompleted processing {len(self.dataset)} wallets")
        print(f"Success rate: {len(self.dataset)/count*100:.2f}%")
        print(f"Time taken: {duration:.2f} seconds")
        
        return df
    
    def save_dataset(self, filename=None):
        """Save dataset to CSV"""
        if not self.dataset:
            raise ValueError("No dataset to save")
            
        if not filename:
            prefix = "mock" if any('private_key' in x for x in self.dataset) else "real"
            filename = f"eth_credit_scores_{prefix}_{time.strftime('%Y%m%d_%H%M')}.csv"
        
        df = pd.DataFrame(self.dataset)
        
        # Don't save private keys in production!
        if 'private_key' in df.columns:
            print("Warning: Private keys included in dataset - not recommended for production!")
            # df = df.drop(columns=['private_key'])  # Uncomment to automatically remove
        
        df.to_csv(filename, index=False)
        print(f"Dataset saved to {filename}")
        
        if self.failed_addresses:
            with open('failed_addresses.txt', 'w') as f:
                f.write('\n'.join(self.failed_addresses))
            print(f"Failed addresses saved to failed_addresses.txt")

if __name__ == "__main__":
    print("""
    Ethereum Credit Scoring Dataset Generator
    ======================================
    Generates wallet datasets with credit scores
    """)
    
    generator = EthereumCreditDatasetGenerator()
    
    # Configure with your Etherscan API key
    generator.configure_etherscan("DMXFNRAC6FEGMGGE68VRWCD6R5EB9RXV6D")
    
    try:
        # Generate dataset with mock data (recommended for testing)
        dataset = generator.generate_dataset(count=500, use_mock_data=True, workers=4)
        
        # Save results
        generator.save_dataset()
        
        # Show stats
        print("\nDataset Preview:")
        print(dataset.head())
        
        print("\nScore Distribution:")
        print(dataset['score'].describe())
        
    except Exception as e:
        print(f"\nError: {str(e)}")
        if generator.dataset:
            print("Attempting to save partial dataset...")
            generator.save_dataset("partial_dataset.csv")
import random
import time
from tqdm import tqdm
import pandas as pd
from eth_utils import to_checksum_address
from eth_account import Account
from concurrent.futures import ThreadPoolExecutor, as_completed
from Eth_Credit_Score import EthereumCreditScorer # Importing your existing code

class EthereumCreditDatasetGenerator:
    def __init__(self):
        self.scorer = EthereumCreditScorer()
        self.wallets = []
        self.dataset = []
        self.failed_addresses = []
        
    def generate_random_wallets(self, count=10000):
        """Generate random Ethereum wallet addresses"""
        print(f"Generating {count} random wallet addresses...")
        self.wallets = [Account.create() for _ in range(count)]
        return self.wallets
    
    def get_realistic_features(self):
        """Generate realistic wallet features based on actual blockchain patterns"""
        return {
            'account_age_days': random.choice([
                *[random.uniform(1, 30) for _ in range(20)],  # 20% new wallets
                *[random.uniform(30, 180) for _ in range(30)],  # 30% 1-6 months
                *[random.uniform(180, 365) for _ in range(25)],  # 25% 6-12 months
                *[random.uniform(365, 365*3) for _ in range(15)],  # 15% 1-3 years
                *[random.uniform(365*3, 365*5) for _ in range(10)]  # 10% 3-5 years
            ]),
            'eth_balance': random.choice([
                *[random.uniform(0, 0.1) for _ in range(40)],  # 40% < 0.1 ETH
                *[random.uniform(0.1, 1) for _ in range(30)],  # 30% 0.1-1 ETH
                *[random.uniform(1, 10) for _ in range(20)],  # 20% 1-10 ETH
                *[random.uniform(10, 100) for _ in range(8)],  # 8% 10-100 ETH
                *[random.uniform(100, 1000) for _ in range(2)]  # 2% 100-1000 ETH
            ]),
            'tx_count': random.choice([
                *[random.randint(0, 10) for _ in range(20)],  # 20% 0-10 txs
                *[random.randint(10, 100) for _ in range(30)],  # 30% 10-100 txs
                *[random.randint(100, 1000) for _ in range(30)],  # 30% 100-1000 txs
                *[random.randint(1000, 10000) for _ in range(15)],  # 15% 1000-10k txs
                *[random.randint(10000, 100000) for _ in range(5)]  # 5% 10k-100k txs
            ]),
            'tx_success_rate': random.choice([
                *[random.uniform(0.9, 1.0) for _ in range(70)],  # 70% high success
                *[random.uniform(0.7, 0.9) for _ in range(20)],  # 20% medium
                *[random.uniform(0, 0.7) for _ in range(10)]  # 10% low
            ]),
            'malicious_score': random.choice([
                *[0 for _ in range(90)],  # 90% clean
                *[random.uniform(0.1, 0.5) for _ in range(8)],  # 8% some risk
                *[random.uniform(0.5, 1.0) for _ in range(2)]  # 2% high risk
            ]),
            'mixer_score': random.choice([
                *[0 for _ in range(95)],  # 95% clean
                *[random.uniform(0.1, 0.5) for _ in range(4)],  # 4% some usage
                *[random.uniform(0.5, 1.0) for _ in range(1)]  # 1% heavy usage
            ])
        }
    
    def generate_mock_wallet_data(self, wallet):
        """Generate mock wallet data with realistic distributions"""
        features = self.get_realistic_features()
        
        # Calculate some derived features
        defi_interactions = int(features['tx_count'] * random.uniform(0.01, 0.2))
        token_diversity = int(features['tx_count'] * random.uniform(0.005, 0.1))
        
        # Delegation features - only for wallets with decent scores
        base_score = (
            300 + 
            (features['account_age_days'] / 365 * 10) + 
            (features['eth_balance'] * 5) +
            (features['tx_count'] * 0.02) +
            (features['tx_success_rate'] * 100) +
            (defi_interactions * 2) +
            (token_diversity * 5) -
            (features['malicious_score'] * 200) -
            (features['mixer_score'] * 150)
        )
        
        has_delegations = base_score > 500 and random.random() < 0.3
        
        return {
            'address': wallet.address,
            'account_age_days': features['account_age_days'],
            'eth_balance': features['eth_balance'],
            'tx_count': features['tx_count'],
            'tx_success_rate': features['tx_success_rate'],
            'defi_interactions': defi_interactions,
            'token_diversity': token_diversity,
            'malicious_score': features['malicious_score'],
            'mixer_score': features['mixer_score'],
            'delegation_capacity': min(features['eth_balance'] * 0.1, 100),
            'staked_collateral': min(features['eth_balance'] * 0.05, 50),
            'reputation_score': random.uniform(0, 1) if has_delegations else 0,
            'delegations_given': random.randint(0, 5) if has_delegations else 0,
            'delegations_received': random.randint(0, 3) if has_delegations else 0,
            'delegation_success_rate': random.uniform(0.7, 1.0) if has_delegations else 0,
            'zkp_usage': 1 if has_delegations and random.random() < 0.2 else 0,
            '_security_data': {
                'malicious': {
                    'count': int(features['malicious_score'] * 3),
                    'details': [],
                    'score': features['malicious_score']
                },
                'mixer': {
                    'count': int(features['mixer_score'] * 2),
                    'transactions': [],
                    'score': features['mixer_score']
                }
            },
            '_delegation_data': {
                'delegations_out': [{
                    'delegatee': to_checksum_address(Account.create().address),
                    'amount': random.randint(10, 50),
                    'timestamp': int(time.time()) - random.randint(0, 86400*30)
                } for _ in range(random.randint(0, 3))] if has_delegations else [],
                'delegations_in': [{
                    'delegator': to_checksum_address(Account.create().address),
                    'amount': random.randint(10, 50),
                    'timestamp': int(time.time()) - random.randint(0, 86400*30)
                } for _ in range(random.randint(0, 2))] if has_delegations else []
            }
        }
    
    def process_wallet(self, wallet, use_mock_data=True):
        """Process a single wallet to generate credit data"""
        try:
            if use_mock_data:
                wallet_data = self.generate_mock_wallet_data(wallet)
            else:
                wallet_data = self.scorer.fetch_wallet_data(wallet.address)
            
            if wallet_data:
                score = self.scorer.predict_score(wallet_data)
                
                # Create dataset record
                record = {
                    'address': wallet.address,
                    'private_key': wallet.key.hex() if use_mock_data else 'N/A',
                    'score': score,
                    **{k: v for k, v in wallet_data.items() if not k.startswith('_')}
                }
                
                # Flatten security data
                record.update({
                    'malicious_interactions': wallet_data['_security_data']['malicious']['count'],
                    'mixer_transactions': wallet_data['_security_data']['mixer']['count'],
                    'delegations_out': len(wallet_data['_delegation_data']['delegations_out']),
                    'delegations_in': len(wallet_data['_delegation_data']['delegations_in'])
                })
                
                return record
            return None
        except Exception as e:
            self.failed_addresses.append(wallet.address)
            return None
    
    def generate_dataset(self, count=10000, use_mock_data=True, workers=8):
        """Generate the complete dataset"""
        start_time = time.time()
        
        # Generate or load wallets
        if not self.wallets:
            self.generate_random_wallets(count)
        
        print(f"Processing {len(self.wallets)} wallets ({'mock' if use_mock_data else 'real'} data)...")
        
        # Process wallets in parallel
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = [executor.submit(self.process_wallet, wallet, use_mock_data) 
                      for wallet in self.wallets]
            
            for future in tqdm(as_completed(futures), total=len(futures)):
                result = future.result()
                if result:
                    self.dataset.append(result)
        
        # Create DataFrame
        df = pd.DataFrame(self.dataset)
        
        # Calculate processing stats
        duration = time.time() - start_time
        success_rate = len(self.dataset) / count * 100
        
        print(f"\nDataset generation complete!")
        print(f"• Total wallets processed: {count}")
        print(f"• Successful records: {len(self.dataset)} ({success_rate:.2f}%)")
        print(f"• Failed addresses: {len(self.failed_addresses)}")
        print(f"• Processing time: {duration:.2f} seconds")
        print(f"• Avg time per wallet: {duration/count:.4f} seconds")
        
        return df
    
    def save_dataset(self, filename='ethereum_credit_scores.csv'):
        """Save the generated dataset to CSV"""
        if not self.dataset:
            print("No dataset to save. Generate dataset first.")
            return
        
        df = pd.DataFrame(self.dataset)
        
        # Don't save private keys in real-world scenario
        if 'private_key' in df.columns:
            df = df.drop(columns=['private_key'])
        
        df.to_csv(filename, index=False)
        print(f"Dataset saved to {filename}")
        
        # Save failed addresses separately
        if self.failed_addresses:
            with open('failed_addresses.txt', 'w') as f:
                f.write('\n'.join(self.failed_addresses))
            print(f"Failed addresses saved to failed_addresses.txt")

if __name__ == "__main__":
    print("""
    Ethereum Credit Scoring Dataset Generator
    =======================================
    This tool generates a dataset of 10,000 Ethereum wallets with credit scores
    using the RBDCS (Reputation-Backed Delegated Credit Scoring) system.
    """)
    
    generator = EthereumCreditDatasetGenerator()
    
    # Configuration
    SAMPLE_SIZE = 10000
    USE_MOCK_DATA = True  # Set to False to use real blockchain data (much slower)
    WORKERS = 8  # Number of parallel threads
    
    # Generate dataset
    dataset = generator.generate_dataset(
        count=SAMPLE_SIZE,
        use_mock_data=USE_MOCK_DATA,
        workers=WORKERS
    )
    
    # Save results
    generator.save_dataset()
    
    # Show sample data
    print("\nSample data:")
    print(dataset.head())
    
    # Basic statistics
    print("\nDataset statistics:")
    print(dataset.describe())
    
    # Score distribution
    print("\nCredit Score Distribution:")
    print(dataset['score'].value_counts(bins=10, sort=False))
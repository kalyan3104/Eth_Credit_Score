import requests
from web3 import Web3
import time
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import StandardScaler
from datetime import datetime
import warnings
warnings.filterwarnings("ignore")

# ========== CONFIGURATION ========== #
INFURA_URL = "https://mainnet.infura.io/v3/af423287a003493f862148da13f7e798"
ETHERSCAN_API_KEY = "DMXFNRAC6FEGMGGE68VRWCD6R5EB9RXV6D"
MAX_RETRIES = 3
RETRY_DELAY = 2

# ========== SECURITY DATABASES ========== #
MALICIOUS_DB = {
    "0x8576acc5c05d6ce88f4e49bf65bdf0c62f91353c": "Cryptopia Hacker",
    "0x098b716b8aaf21512996dc57eb0615e2383e2f96": "Fake_Phishing1723",
    "0x19aa5fe80d33a56d56c78e82ea5e50e5d80b4dff": "Twister Money Mixer",
    "0x1da5821544e25c636c1417ba96ade4cf6d2f9b5a": "Tornado Cash"
}

MIXER_ADDRESSES = [
    "0x1da5821544e25c636c1417ba96ade4cf6d2f9b5a",  # Tornado Cash
    "0x19aa5fe80d33a56d56c78e82ea5e50e5d80b4dff"   # Twister Money
]

REPUTABLE_CONTRACTS = {
    "0x7a250d5630B4cF539739dF2C5dAcb4c659F2488D": "Uniswap V2 Router",
    "0x3d9819210A31b4961b30EF54bE2aeD79B9c9Cd3B": "Compound",
    "0x7d2768dE32b0b80b7a3454c06BdAc94A69DDc7A9": "Aave V2"
}

# ========== DELEGATION CONTRACT CONFIG ========== #
DELEGATION_CONTRACT = {
    "address": "0x1234567890123456789012345678901234567890",  # Placeholder address
    "abi": [
        {
            "inputs": [
                {"internalType": "address", "name": "delegatee", "type": "address"},
                {"internalType": "uint256", "name": "amount", "type": "uint256"}
            ],
            "name": "delegateCredit",
            "outputs": [],
            "stateMutability": "nonpayable",
            "type": "function"
        },
        {
            "inputs": [{"internalType": "address", "name": "delegator", "type": "address"}],
            "name": "getDelegationsByDelegator",
            "outputs": [
                {
                    "components": [
                        {"internalType": "address", "name": "delegatee", "type": "address"},
                        {"internalType": "uint256", "name": "amount", "type": "uint256"},
                        {"internalType": "uint256", "name": "timestamp", "type": "uint256"}
                    ],
                    "internalType": "struct DelegationManager.Delegation[]",
                    "name": "",
                    "type": "tuple[]"
                }
            ],
            "stateMutability": "view",
            "type": "function"
        },
        {
            "inputs": [{"internalType": "address", "name": "delegatee", "type": "address"}],
            "name": "getDelegationsByDelegatee",
            "outputs": [
                {
                    "components": [
                        {"internalType": "address", "name": "delegator", "type": "address"},
                        {"internalType": "uint256", "name": "amount", "type": "uint256"},
                        {"internalType": "uint256", "name": "timestamp", "type": "uint256"}
                    ],
                    "internalType": "struct DelegationManager.Delegation[]",
                    "name": "",
                    "type": "tuple[]"
                }
            ],
            "stateMutability": "view",
            "type": "function"
        }
    ]
}

# ========== WEB3 INITIALIZATION ========== #
web3 = Web3(Web3.HTTPProvider(INFURA_URL))

# Connection check with modern Web3 syntax
try:
    if not web3.is_connected():
        raise ConnectionError("❌ Not connected to Ethereum node")
    block = web3.eth.block_number  # Updated property name
    print(f"✅ Successfully connected to Ethereum node (Block: {block})")
except Exception as e:
    raise ConnectionError(f"❌ Failed to connect to Ethereum node: {str(e)}")

class DelegationManager:
    def __init__(self, web3):
        self.web3 = web3
        try:
            self.contract = self.web3.eth.contract(
                address=DELEGATION_CONTRACT["address"],
                abi=DELEGATION_CONTRACT["abi"]
            )
        except Exception as e:
            print(f"⚠️ Warning: Could not connect to delegation contract. Using mock mode. Error: {str(e)}")
            self.contract = None
            self.mock_delegations = {}
    
    def delegate_credit(self, delegator, delegatee, amount, private_key):
        """Delegate a portion of credit score to another address"""
        if self.contract is None:
            # Mock mode - store delegation in memory
            if delegator not in self.mock_delegations:
                self.mock_delegations[delegator] = []
            self.mock_delegations[delegator].append({
                'delegatee': delegatee,
                'amount': amount,
                'timestamp': int(time.time())
            })
            return "0xmocktxhash" + str(int(time.time()))
        
        try:
            # Build transaction
            tx = self.contract.functions.delegateCredit(
                delegatee,
                amount
            ).build_transaction({
                'from': delegator,
                'nonce': self.web3.eth.get_transaction_count(delegator),
                'gas': 200000,
                'gasPrice': self.web3.eth.gas_price
            })
            
            # Sign and send
            signed_tx = self.web3.eth.account.sign_transaction(tx, private_key)
            tx_hash = self.web3.eth.send_raw_transaction(signed_tx.rawTransaction)
            
            return tx_hash.hex()
        except Exception as e:
            print(f"Delegation failed: {str(e)}")
            return None
    
    def get_delegations(self, address):
        """Get all delegations for an address"""
        if self.contract is None:
            # Mock mode - return mock delegations
            delegations_out = self.mock_delegations.get(address, [])
            delegations_in = []
            for delegator, dels in self.mock_delegations.items():
                for d in dels:
                    if d['delegatee'].lower() == address.lower():
                        delegations_in.append({
                            'delegator': delegator,
                            'amount': d['amount'],
                            'timestamp': d['timestamp']
                        })
            return {
                'delegations_out': delegations_out,
                'delegations_in': delegations_in
            }
        
        try:
            # Get delegations where address is delegator
            delegations_out = self.contract.functions.getDelegationsByDelegator(address).call()
            
            # Get delegations where address is delegatee
            delegations_in = self.contract.functions.getDelegationsByDelegatee(address).call()
            
            return {
                'delegations_out': delegations_out,
                'delegations_in': delegations_in
            }
        except Exception as e:
            print(f"Failed to get delegations: {str(e)}")
            return None
    
    def calculate_boosted_score(self, original_score, delegations):
        """Calculate boosted score based on incoming delegations"""
        if not delegations or not delegations['delegations_in']:
            return original_score
        
        total_delegated = sum(d['amount'] for d in delegations['delegations_in'])
        max_boost = min(total_delegated * 0.1, 200)  # Max 200 point boost
        boosted_score = original_score + max_boost
        
        return min(boosted_score, 1000)  # Cap at 1000

class EthereumCreditScorer:
    def __init__(self):
        self.web3 = web3
        self.scaler = StandardScaler()
        self.model = self.initialize_model()
        self.delegation_manager = DelegationManager(web3)
        
    def initialize_model(self):
        """Create and train the credit scoring model with RBDCS features"""
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        
        # Enhanced synthetic data generation with RBDCS features
        num_samples = 5000
        
        # Base features
        data = {
            'account_age_days': np.random.gamma(shape=2, scale=200, size=num_samples),
            'eth_balance': np.random.exponential(scale=5, size=num_samples),
            'tx_count': np.random.poisson(lam=1000, size=num_samples),
            'tx_success_rate': np.random.beta(a=9, b=1, size=num_samples),
            'defi_interactions': np.random.poisson(lam=25, size=num_samples),
            'token_diversity': np.random.poisson(lam=8, size=num_samples),
            'malicious_score': np.random.beta(a=1, b=9, size=num_samples),
            'mixer_score': np.random.beta(a=1, b=9, size=num_samples),
            
            # RBDCS-specific features
            'delegation_capacity': np.random.uniform(0, 100, size=num_samples),
            'staked_collateral': np.random.exponential(scale=2, size=num_samples),
            'reputation_score': np.random.beta(a=2, b=2, size=num_samples),
            'delegations_given': np.random.poisson(lam=3, size=num_samples),
            'delegations_received': np.random.poisson(lam=2, size=num_samples),
            'delegation_success_rate': np.clip(
                np.random.beta(a=9, b=1, size=num_samples) * np.random.normal(1, 0.1, size=num_samples),
                0, 1
            ),
            'zkp_usage': np.random.binomial(1, 0.3, size=num_samples)
        }
        
        # Generate credit scores with RBDCS boosts
        base_score = (
            300 + 
            (data['account_age_days'] / 365 * 10) + 
            (np.log1p(data['eth_balance']) * 50) +
            (np.log1p(data['tx_count']) * 20) +
            (data['tx_success_rate'] * 100) +
            (data['defi_interactions'] * 2) +
            (data['token_diversity'] * 5) -
            (data['malicious_score'] * 200) -
            (data['mixer_score'] * 150)
        )
        
        reputation_boost = (
            (data['reputation_score'] * 50) +
            (data['delegation_success_rate'] * 30) +
            (np.log1p(data['delegations_given']) * 10) +
            (np.log1p(data['delegations_received']) * 5)
        )
        
        data['credit_score'] = np.clip(base_score + reputation_boost, 300, 1000)
        
        # Create DataFrame
        df = pd.DataFrame(data)
        
        # Select features for training
        features = [
            'account_age_days', 'eth_balance', 'tx_count', 'tx_success_rate',
            'defi_interactions', 'token_diversity', 'malicious_score', 'mixer_score',
            'delegation_capacity', 'staked_collateral', 'reputation_score',
            'delegations_given', 'delegations_received', 'delegation_success_rate'
        ]
        
        X = df[features]
        y = df['credit_score']
        
        # Train/test split and scaling
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        X_train = self.scaler.fit_transform(X_train)
        model.fit(X_train, y_train)
        
        # Evaluate model
        train_score = model.score(X_train, y_train)
        test_score = model.score(self.scaler.transform(X_test), y_test)
        print(f"Model trained. R²: Train={train_score:.2f}, Test={test_score:.2f}")
        return model

    def safe_api_request(self, url):
        """Make API request with retry logic"""
        for attempt in range(MAX_RETRIES):
            try:
                response = requests.get(url, timeout=15)
                response.raise_for_status()
                return response.json()
            except Exception as e:
                if attempt < MAX_RETRIES - 1:
                    time.sleep(RETRY_DELAY)
                else:
                    raise Exception(f"API request failed after {MAX_RETRIES} attempts: {str(e)}")

    def get_transactions(self, address):
        """Get transaction history"""
        url = f"https://api.etherscan.io/api?module=account&action=txlist&address={address}&startblock=0&endblock=99999999&sort=asc&apikey={ETHERSCAN_API_KEY}"
        data = self.safe_api_request(url)
        return data.get('result', [])

    def check_security_risks(self, transactions):
        """Analyze transactions for security risks"""
        malicious = {
            'count': 0,
            'details': [],
            'score': 0
        }
        
        mixer = {
            'count': 0,
            'transactions': [],
            'score': 0
        }
        
        for tx in transactions:
            # Check for malicious addresses
            if tx.get('to') in MALICIOUS_DB or tx.get('from') in MALICIOUS_DB:
                malicious['count'] += 1
                malicious['details'].append({
                    'hash': tx['hash'],
                    'address': tx['to'] if tx.get('to') in MALICIOUS_DB else tx['from'],
                    'label': MALICIOUS_DB[tx['to'] if tx.get('to') in MALICIOUS_DB else tx['from']],
                    'value': float(tx.get('value', 0)) / 1e18,
                    'timestamp': tx.get('timeStamp')
                })
            
            # Check for mixer transactions
            if tx.get('to') in MIXER_ADDRESSES or tx.get('from') in MIXER_ADDRESSES:
                mixer['count'] += 1
                mixer['transactions'].append({
                    'hash': tx['hash'],
                    'mixer': tx['to'] if tx.get('to') in MIXER_ADDRESSES else tx['from'],
                    'value': float(tx.get('value', 0)) / 1e18,
                    'timestamp': tx.get('timeStamp')
                })
        
        # Calculate risk scores (capped at 1.0)
        malicious['score'] = min(malicious['count'] / 3, 1.0) 
        mixer['score'] = min(mixer['count'] / 2, 1.0)  
        
        return malicious, mixer

    def fetch_wallet_data(self, address):
        """Collect comprehensive wallet data"""
        print("\n🔍 Collecting blockchain data...")
        
        try:
            # Basic on-chain data
            balance = self.web3.eth.get_balance(address)
            tx_count = self.web3.eth.get_transaction_count(address)
            txs = self.get_transactions(address)
            
            # Calculate derived metrics
            first_tx = min([int(tx['timeStamp']) for tx in txs]) if txs else time.time()
            tx_success_rate = sum(1 for tx in txs if tx.get('isError') == '0') / len(txs) if txs else 1.0
            token_diversity = len(set(tx.get('contractAddress', '') for tx in txs if tx.get('contractAddress')))
            
            # Security analysis
            malicious, mixer = self.check_security_risks(txs)
            
            # Get delegation data
            delegations = self.delegation_manager.get_delegations(address)
            
            # Calculate RBDCS metrics
            delegations_given = len(delegations['delegations_out']) if delegations else 0
            delegations_received = len(delegations['delegations_in']) if delegations else 0
            delegation_success_rate = 0.9 if delegations_given > 0 else 0  # Placeholder
            
            return {
                'account_age_days': (time.time() - first_tx) / 86400,
                'eth_balance': float(balance) / 1e18,
                'tx_count': tx_count,
                'tx_success_rate': tx_success_rate,
                'defi_interactions': sum(1 for tx in txs if tx.get('to') in REPUTABLE_CONTRACTS),
                'token_diversity': token_diversity,
                'malicious_score': malicious['score'],
                'mixer_score': mixer['score'],
                'delegation_capacity': min(float(balance) / 1e18 * 0.1, 100),  # 10% of balance
                'staked_collateral': min(float(balance) / 1e18 * 0.05, 50),    # 5% of balance
                'reputation_score': min((delegations_given + delegations_received) / 10, 1),
                'delegations_given': delegations_given,
                'delegations_received': delegations_received,
                'delegation_success_rate': delegation_success_rate,
                'zkp_usage': 1 if delegations and delegations['delegations_in'] else 0,
                '_security_data': {
                    'malicious': malicious,
                    'mixer': mixer
                },
                '_delegation_data': delegations
            }
            
        except Exception as e:
            print(f"❌ Error analyzing wallet: {str(e)}")
            return None

    def predict_score(self, wallet_data):
        """Predict credit score with proper feature scaling"""
        if wallet_data is None:
            return 300  # Minimum score if analysis fails
            
        # Prepare features DataFrame
        features = pd.DataFrame([[
            wallet_data['account_age_days'],
            wallet_data['eth_balance'],
            wallet_data['tx_count'],
            wallet_data['tx_success_rate'],
            wallet_data['defi_interactions'],
            wallet_data['token_diversity'],
            wallet_data['malicious_score'],
            wallet_data['mixer_score'],
            wallet_data['delegation_capacity'],
            wallet_data['staked_collateral'],
            wallet_data['reputation_score'],
            wallet_data['delegations_given'],
            wallet_data['delegations_received'],
            wallet_data['delegation_success_rate']
        ]], columns=[
            'account_age_days',
            'eth_balance',
            'tx_count',
            'tx_success_rate',
            'defi_interactions',
            'token_diversity',
            'malicious_score',
            'mixer_score',
            'delegation_capacity',
            'staked_collateral',
            'reputation_score',
            'delegations_given',
            'delegations_received',
            'delegation_success_rate'
        ])
        
        # Scale features and predict
        features_scaled = self.scaler.transform(features)
        score = self.model.predict(features_scaled)[0]
        
        # Apply delegation boost if available
        if '_delegation_data' in wallet_data:
            score = self.delegation_manager.calculate_boosted_score(score, wallet_data['_delegation_data'])
        
        return max(300, min(1000, score))  # Ensure score is within bounds

    def generate_report(self, address, wallet_data, score):
        """Generate comprehensive credit report with delegation info"""
        print("\n" + "="*60)
        print("🔍 Ethereum Credit Score Report")
        print("="*60)
        print(f"Address: {address}")
        
        # Account Overview
        print("\n=== ACCOUNT OVERVIEW ===")
        print(f"• Age: {wallet_data['account_age_days']:.1f} days")
        print(f"• Balance: {wallet_data['eth_balance']:.4f} ETH")
        print(f"• Transactions: {wallet_data['tx_count']}")
        print(f"• Success Rate: {wallet_data['tx_success_rate']:.2%}")
        print(f"• Token Diversity: {wallet_data['token_diversity']}")
        print(f"• DeFi Interactions: {wallet_data['defi_interactions']}")
        
        # Security Analysis
        print("\n=== SECURITY ANALYSIS ===")
        malicious = wallet_data['_security_data']['malicious']
        mixer = wallet_data['_security_data']['mixer']
        
        print(f"• Malicious Interactions: {malicious['count']}")
        if malicious['count'] > 0:
            print("  Detected:")
            for i, item in enumerate(malicious['details'][:3]):  # Show first 3 for brevity
                print(f"  {i+1}. {item['label']} ({item['address']})")
                print(f"     TX: {item['hash']} | Value: {item['value']:.4f} ETH")
        
        print(f"\n• Mixer Transactions: {mixer['count']}")
        if mixer['count'] > 0:
            print("  Detected:")
            for i, item in enumerate(mixer['transactions'][:3]):
                print(f"  {i+1}. {item['mixer']}")
                print(f"     TX: {item['hash']} | Value: {item['value']:.4f} ETH")
        
        # RBDCS Features
        print("\n=== REPUTATION-BACKED DELEGATION ===")
        print(f"• Delegation Capacity: {wallet_data['delegation_capacity']:.2f} points")
        print(f"• Staked Collateral: {wallet_data['staked_collateral']:.4f} ETH")
        print(f"• Reputation Score: {wallet_data['reputation_score']:.2f}/1.0")
        print(f"• Successful Delegations: {wallet_data['delegation_success_rate']:.2%}")
        print(f"• Uses ZKP: {'Yes' if wallet_data['zkp_usage'] else 'No'}")
        
        # Delegation Information
        if '_delegation_data' in wallet_data and wallet_data['_delegation_data']:
            delegations = wallet_data['_delegation_data']
            print("\n=== DELEGATION NETWORK ===")
            if delegations['delegations_out']:
                print(f"• You are delegating credit to {len(delegations['delegations_out'])} addresses")
                for i, d in enumerate(delegations['delegations_out'][:3]):  # Show first 3
                    print(f"  {i+1}. To: {d['delegatee']} - Amount: {d['amount']} points")
            
            if delegations['delegations_in']:
                print(f"\n• You are receiving credit from {len(delegations['delegations_in'])} addresses")
                for i, d in enumerate(delegations['delegations_in'][:3]):
                    print(f"  {i+1}. From: {d['delegator']} - Amount: {d['amount']} points")
        
        # Credit Score
        print("\n" + "="*60)
        print(f"💳 FINAL CREDIT SCORE: {score:.0f}/1000")
        
        # Score interpretation
        if score >= 850:
            print("💎 Excellent - Exceptional creditworthiness")
        elif score >= 700:
            print("👍 Very Good - Strong financial history")
        elif score >= 550:
            print("🆗 Good - Reliable with minor risks")
        elif score >= 400:
            print("⚠️ Fair - Elevated risk factors")
        else:
            print("❌ Poor - High risk profile")
        
        print("="*60)

if __name__ == "__main__":
    print("""                                                                  
    Ethereum Credit Scoring System with RBDCS v0.3
    """)
    
    scorer = EthereumCreditScorer()
    
    # Get valid Ethereum address from user
    while True:
        address = input("\nEnter Ethereum wallet address (0x...): ").strip()
        if web3.is_address(address):
            address = web3.to_checksum_address(address)
            break
        print("❌ Invalid address format. Please try again.")
    
    # Analyze wallet and generate report
    wallet_data = scorer.fetch_wallet_data(address)
    
    if wallet_data:
        # Get score (includes RBDCS boosts)
        score = scorer.predict_score(wallet_data)
        
        # Generate enhanced report
        scorer.generate_report(address, wallet_data, score)
        
        # Show delegation options if score is high enough
        if score >= 700 and wallet_data['delegation_capacity'] > 10:
            print("\nWould you like to delegate some of your credit score to another address?")
            choice = input("(Y)es / (N)o: ").strip().lower()
            
            if choice == 'y':
                delegatee = input("Enter address to delegate to (0x...): ").strip()
                if web3.is_address(delegatee):
                    delegatee = web3.to_checksum_address(delegatee)
                    max_amount = min(wallet_data['delegation_capacity'], 100)
                    amount = int(input(f"Enter amount to delegate (10-{max_amount}): "))
                    private_key = input("Enter your private key (will not be stored): ").strip()
                    
                    tx_hash = scorer.delegation_manager.delegate_credit(
                        address,
                        delegatee,
                        amount,
                        private_key
                    )
                    
                    if tx_hash:
                        print(f"✅ Delegation successful! TX Hash: {tx_hash}")
                        print("Please wait a few minutes for the delegation to be reflected in your score.")
                    else:
                        print("❌ Delegation failed")
                else:
                    print("❌ Invalid delegatee address")
    else:
        print("❌ Failed to analyze wallet. Please check the address and try again.")
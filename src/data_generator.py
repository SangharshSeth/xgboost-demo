"""
Indian Bank Transaction Data Generator

Generates realistic transaction data for fraud detection model training.
Features:
- 500 customers with diverse profiles (salaried, self-employed, students, businesses, retired)
- 12 months of transaction history (~150-180K transactions)
- Proper balance tracking (every transaction affects running balance)
- 0.1% fraud rate with realistic fraud patterns
- EMI debits, salary credits, UPI, cards, ATM withdrawals
"""

import random
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Optional
import pandas as pd
import numpy as np


# ============================================================================
# ENUMS AND CONSTANTS
# ============================================================================

class CustomerType(Enum):
    SALARIED = "salaried"
    SELF_EMPLOYED = "self_employed"
    STUDENT = "student"
    SMALL_BUSINESS = "small_business"
    RETIRED = "retired"


class TransactionType(Enum):
    UPI = "UPI"
    DEBIT_CARD = "DEBIT_CARD"
    CREDIT_CARD = "CREDIT_CARD"
    NEFT = "NEFT"
    IMPS = "IMPS"
    ATM = "ATM"
    SALARY = "SALARY"
    PENSION = "PENSION"
    EMI = "EMI"
    REFUND = "REFUND"
    PARENT_TRANSFER = "PARENT_TRANSFER"  # For students


class TransactionStatus(Enum):
    SUCCESS = "SUCCESS"
    FAILED = "FAILED"
    PENDING = "PENDING"


# Popular Indian merchants by category
MERCHANTS = {
    "grocery": [
        ("BigBasket", "MCC_5411"), ("Blinkit", "MCC_5411"), ("Zepto", "MCC_5411"),
        ("DMart", "MCC_5411"), ("Reliance Fresh", "MCC_5411"), ("More", "MCC_5411"),
        ("Local Kirana Store", "MCC_5411"), ("Spencer's", "MCC_5411")
    ],
    "food_delivery": [
        ("Swiggy", "MCC_5812"), ("Zomato", "MCC_5812"), ("Dominos", "MCC_5812"),
        ("McDonalds", "MCC_5814"), ("KFC", "MCC_5814"), ("Pizza Hut", "MCC_5812")
    ],
    "ecommerce": [
        ("Amazon", "MCC_5311"), ("Flipkart", "MCC_5311"), ("Myntra", "MCC_5651"),
        ("Ajio", "MCC_5651"), ("Nykaa", "MCC_5977"), ("Meesho", "MCC_5311"),
        ("Tata Cliq", "MCC_5311"), ("Snapdeal", "MCC_5311")
    ],
    "fuel": [
        ("Indian Oil", "MCC_5541"), ("HP Petrol", "MCC_5541"), ("Bharat Petroleum", "MCC_5541"),
        ("Shell", "MCC_5541"), ("Reliance Petroleum", "MCC_5541")
    ],
    "utilities": [
        ("Electricity Board", "MCC_4900"), ("Water Board", "MCC_4900"),
        ("Gas Agency", "MCC_4900"), ("Broadband", "MCC_4814"), ("Mobile Recharge", "MCC_4814"),
        ("DTH Recharge", "MCC_4899")
    ],
    "travel": [
        ("IRCTC", "MCC_4112"), ("MakeMyTrip", "MCC_4722"), ("Goibibo", "MCC_4722"),
        ("Ola", "MCC_4121"), ("Uber", "MCC_4121"), ("Rapido", "MCC_4121"),
        ("RedBus", "MCC_4131"), ("IndiGo", "MCC_4511"), ("SpiceJet", "MCC_4511")
    ],
    "entertainment": [
        ("Netflix", "MCC_7832"), ("Amazon Prime", "MCC_7832"), ("Hotstar", "MCC_7832"),
        ("BookMyShow", "MCC_7832"), ("Spotify", "MCC_7832"), ("YouTube Premium", "MCC_7832"),
        ("PVR Cinemas", "MCC_7832"), ("INOX", "MCC_7832")
    ],
    "healthcare": [
        ("Apollo Pharmacy", "MCC_5912"), ("MedPlus", "MCC_5912"), ("1mg", "MCC_5912"),
        ("PharmEasy", "MCC_5912"), ("Practo", "MCC_8011"), ("Hospital", "MCC_8062")
    ],
    "education": [
        ("School Fees", "MCC_8211"), ("College Fees", "MCC_8220"), ("Byju's", "MCC_8299"),
        ("Unacademy", "MCC_8299"), ("Coursera", "MCC_8299"), ("Udemy", "MCC_8299")
    ],
    "rent": [
        ("Landlord", "MCC_6513"), ("NoBroker", "MCC_6513"), ("Housing Society", "MCC_6513")
    ],
    "emi": [
        ("HDFC Bank EMI", "MCC_6012"), ("ICICI Bank EMI", "MCC_6012"), ("SBI EMI", "MCC_6012"),
        ("Bajaj Finserv EMI", "MCC_6012"), ("Home Loan EMI", "MCC_6012"), ("Car Loan EMI", "MCC_6012")
    ],
    "investment": [
        ("Zerodha", "MCC_6211"), ("Groww", "MCC_6211"), ("Upstox", "MCC_6211"),
        ("SIP Mutual Fund", "MCC_6211")
    ],
    "p2p_transfer": [
        ("Friend Transfer", "MCC_4829"), ("Family Transfer", "MCC_4829"),
        ("Business Payment", "MCC_4829")
    ]
}

# Indian cities with tier classification
CITIES = {
    "tier1": ["Mumbai", "Delhi", "Bangalore", "Chennai", "Hyderabad", "Kolkata", "Pune", "Ahmedabad"],
    "tier2": ["Jaipur", "Lucknow", "Kanpur", "Nagpur", "Indore", "Bhopal", "Visakhapatnam", "Patna", 
              "Vadodara", "Coimbatore", "Ludhiana", "Agra", "Nashik", "Ranchi", "Guwahati"],
    "tier3": ["Dehradun", "Shimla", "Mysore", "Mangalore", "Trichy", "Raipur", "Jabalpur", 
              "Gwalior", "Jodhpur", "Amritsar", "Allahabad", "Varanasi", "Srinagar", "Jammu"]
}


# ============================================================================
# DATA CLASSES
# ============================================================================

@dataclass
class Customer:
    customer_id: str
    customer_type: CustomerType
    name: str
    age: int
    city: str
    city_tier: str
    monthly_income: float
    initial_balance: float
    current_balance: float
    account_created: datetime
    has_emi: bool = False
    emi_amount: float = 0.0
    emi_day: int = 5  # Day of month for EMI debit
    salary_day: int = 1  # Day of month for salary credit
    avg_monthly_transactions: int = 30
    
    # Behavioral patterns
    preferred_categories: list = field(default_factory=list)
    typical_txn_hours: tuple = (9, 22)  # Active hours
    weekend_activity_multiplier: float = 1.0


@dataclass
class Transaction:
    transaction_id: str
    customer_id: str
    timestamp: datetime
    amount: float
    transaction_type: TransactionType
    merchant_name: str
    merchant_category: str
    mcc_code: str
    city: str
    is_credit: bool  # True for credit, False for debit
    balance_before: float
    balance_after: float
    status: TransactionStatus
    device_id: str
    is_fraud: bool = False
    fraud_type: Optional[str] = None


# ============================================================================
# CUSTOMER GENERATOR
# ============================================================================

class CustomerGenerator:
    """Generates realistic Indian customer profiles."""
    
    INDIAN_FIRST_NAMES = [
        "Aarav", "Vivaan", "Aditya", "Vihaan", "Arjun", "Sai", "Reyansh", "Ayaan",
        "Krishna", "Ishaan", "Shaurya", "Atharva", "Advait", "Dhruv", "Kabir",
        "Ananya", "Aadhya", "Myra", "Pari", "Aanya", "Diya", "Pihu", "Prisha",
        "Anvi", "Riya", "Sara", "Navya", "Aashi", "Ira", "Kiara",
        "Ramesh", "Suresh", "Mahesh", "Rajesh", "Amit", "Vijay", "Arun", "Sanjay",
        "Sunita", "Kavita", "Neha", "Pooja", "Priya", "Sneha", "Anjali", "Deepa"
    ]
    
    INDIAN_LAST_NAMES = [
        "Sharma", "Verma", "Gupta", "Singh", "Kumar", "Patel", "Reddy", "Rao",
        "Nair", "Menon", "Iyer", "Joshi", "Kulkarni", "Deshmukh", "Patil",
        "Chatterjee", "Mukherjee", "Banerjee", "Das", "Ghosh", "Sen", "Kapoor",
        "Malhotra", "Khanna", "Bhatia", "Agarwal", "Mittal", "Goel", "Jain"
    ]
    
    # Customer type distribution and parameters
    CUSTOMER_CONFIG = {
        CustomerType.SALARIED: {
            "count": 250,
            "age_range": (23, 58),
            "income_range": (25000, 300000),  # Monthly income
            "balance_range": (50000, 500000),
            "txn_range": (25, 40),
            "categories": ["grocery", "food_delivery", "ecommerce", "fuel", "utilities", "entertainment"],
            "has_emi_prob": 0.4,
            "emi_range": (5000, 50000)
        },
        CustomerType.SELF_EMPLOYED: {
            "count": 100,
            "age_range": (25, 55),
            "income_range": (30000, 500000),
            "balance_range": (100000, 800000),
            "txn_range": (30, 50),
            "categories": ["grocery", "food_delivery", "ecommerce", "fuel", "utilities", "travel", "p2p_transfer"],
            "has_emi_prob": 0.3,
            "emi_range": (10000, 100000)
        },
        CustomerType.STUDENT: {
            "count": 75,
            "age_range": (18, 25),
            "income_range": (5000, 20000),  # Pocket money/part-time
            "balance_range": (10000, 50000),
            "txn_range": (20, 35),
            "categories": ["food_delivery", "ecommerce", "entertainment", "education"],
            "has_emi_prob": 0.05,
            "emi_range": (1000, 5000)
        },
        CustomerType.SMALL_BUSINESS: {
            "count": 50,
            "age_range": (28, 60),
            "income_range": (50000, 800000),
            "balance_range": (200000, 1000000),
            "txn_range": (50, 80),
            "categories": ["utilities", "fuel", "travel", "p2p_transfer", "investment"],
            "has_emi_prob": 0.5,
            "emi_range": (20000, 200000)
        },
        CustomerType.RETIRED: {
            "count": 25,
            "age_range": (58, 75),
            "income_range": (20000, 80000),  # Pension
            "balance_range": (300000, 800000),
            "txn_range": (10, 20),
            "categories": ["grocery", "utilities", "healthcare"],
            "has_emi_prob": 0.1,
            "emi_range": (5000, 20000)
        }
    }
    
    def __init__(self, seed: int = 42):
        random.seed(seed)
        np.random.seed(seed)
        self.start_date = datetime(2024, 1, 1)
        
    def generate_customers(self) -> list[Customer]:
        """Generate all customer profiles."""
        customers = []
        
        for cust_type, config in self.CUSTOMER_CONFIG.items():
            for _ in range(config["count"]):
                customer = self._create_customer(cust_type, config)
                customers.append(customer)
        
        random.shuffle(customers)
        return customers
    
    def _create_customer(self, cust_type: CustomerType, config: dict) -> Customer:
        """Create a single customer profile."""
        # Select city based on customer type (business more likely in tier1)
        if cust_type == CustomerType.SMALL_BUSINESS:
            city_tier = random.choices(["tier1", "tier2", "tier3"], weights=[0.6, 0.3, 0.1])[0]
        elif cust_type == CustomerType.STUDENT:
            city_tier = random.choices(["tier1", "tier2", "tier3"], weights=[0.5, 0.4, 0.1])[0]
        else:
            city_tier = random.choices(["tier1", "tier2", "tier3"], weights=[0.4, 0.4, 0.2])[0]
        
        city = random.choice(CITIES[city_tier])
        
        # Generate income and balance
        monthly_income = random.randint(*config["income_range"])
        initial_balance = random.randint(*config["balance_range"])
        
        # EMI
        has_emi = random.random() < config["has_emi_prob"]
        emi_amount = random.randint(*config["emi_range"]) if has_emi else 0
        
        # Salary/pension day (most common: 1st, but some companies pay on different dates)
        salary_day = random.choices([1, 5, 7, 10, 15, 25, 28], 
                                    weights=[0.4, 0.15, 0.1, 0.1, 0.1, 0.1, 0.05])[0]
        
        # Account created before our transaction period
        account_created = self.start_date - timedelta(days=random.randint(180, 1800))
        
        return Customer(
            customer_id=f"CUST{uuid.uuid4().hex[:8].upper()}",
            customer_type=cust_type,
            name=f"{random.choice(self.INDIAN_FIRST_NAMES)} {random.choice(self.INDIAN_LAST_NAMES)}",
            age=random.randint(*config["age_range"]),
            city=city,
            city_tier=city_tier,
            monthly_income=monthly_income,
            initial_balance=initial_balance,
            current_balance=initial_balance,
            account_created=account_created,
            has_emi=has_emi,
            emi_amount=emi_amount,
            emi_day=random.randint(1, 10),
            salary_day=salary_day,
            avg_monthly_transactions=random.randint(*config["txn_range"]),
            preferred_categories=config["categories"],
            typical_txn_hours=(
                random.randint(7, 10),  # Start hour
                random.randint(20, 23)   # End hour
            ),
            weekend_activity_multiplier=random.uniform(0.8, 1.5)
        )


# ============================================================================
# TRANSACTION GENERATOR
# ============================================================================

class TransactionGenerator:
    """Generates realistic transaction history for customers."""
    
    def __init__(self, seed: int = 42):
        random.seed(seed)
        np.random.seed(seed)
        self.start_date = datetime(2024, 1, 1)
        self.end_date = datetime(2024, 12, 31)
        self.fraud_probability = 0.001  # 0.1% fraud rate
        
    def generate_transactions(self, customers: list[Customer]) -> list[Transaction]:
        """Generate all transactions for all customers."""
        all_transactions = []
        
        for customer in customers:
            customer_txns = self._generate_customer_transactions(customer)
            all_transactions.extend(customer_txns)
        
        # Sort by timestamp
        all_transactions.sort(key=lambda x: x.timestamp)
        
        # Inject fraud patterns
        all_transactions = self._inject_fraud_patterns(all_transactions, customers)
        
        return all_transactions
    
    def _generate_customer_transactions(self, customer: Customer) -> list[Transaction]:
        """Generate 12 months of transactions for a customer."""
        transactions = []
        current_date = self.start_date
        device_id = f"DEV{uuid.uuid4().hex[:8].upper()}"
        
        # Reset customer balance to initial for generation
        customer.current_balance = customer.initial_balance
        
        while current_date <= self.end_date:
            month_txns = self._generate_month_transactions(customer, current_date, device_id)
            transactions.extend(month_txns)
            
            # Move to next month
            if current_date.month == 12:
                current_date = datetime(current_date.year + 1, 1, 1)
            else:
                current_date = datetime(current_date.year, current_date.month + 1, 1)
        
        return transactions
    
    def _generate_month_transactions(self, customer: Customer, month_start: datetime, 
                                     device_id: str) -> list[Transaction]:
        """Generate transactions for a single month."""
        transactions = []
        
        # Determine number of transactions this month (with variation)
        base_txns = customer.avg_monthly_transactions
        monthly_variation = random.uniform(0.7, 1.3)
        num_transactions = int(base_txns * monthly_variation)
        
        # Get month end
        if month_start.month == 12:
            month_end = datetime(month_start.year + 1, 1, 1) - timedelta(seconds=1)
        else:
            month_end = datetime(month_start.year, month_start.month + 1, 1) - timedelta(seconds=1)
        
        # 1. Add salary/pension credit (fixed date)
        if customer.customer_type in [CustomerType.SALARIED, CustomerType.RETIRED]:
            salary_txn = self._create_income_transaction(customer, month_start, device_id)
            if salary_txn:
                transactions.append(salary_txn)
        
        # 2. Add parent transfer for students (around month start)
        if customer.customer_type == CustomerType.STUDENT:
            transfer_txn = self._create_parent_transfer(customer, month_start, device_id)
            if transfer_txn:
                transactions.append(transfer_txn)
        
        # 3. Add self-employed income (irregular)
        if customer.customer_type == CustomerType.SELF_EMPLOYED:
            # 1-3 income transactions per month
            for _ in range(random.randint(1, 3)):
                income_day = random.randint(1, 28)
                income_date = month_start.replace(day=income_day)
                income_txn = self._create_business_income(customer, income_date, device_id)
                if income_txn:
                    transactions.append(income_txn)
        
        # 4. Add EMI debit if applicable
        if customer.has_emi and customer.emi_amount > 0:
            emi_txn = self._create_emi_transaction(customer, month_start, device_id)
            if emi_txn:
                transactions.append(emi_txn)
        
        # 5. Add rent (for salaried/self-employed)
        if customer.customer_type in [CustomerType.SALARIED, CustomerType.SELF_EMPLOYED]:
            if random.random() < 0.6:  # 60% pay rent
                rent_txn = self._create_rent_transaction(customer, month_start, device_id)
                if rent_txn:
                    transactions.append(rent_txn)
        
        # 6. Generate regular spending transactions
        remaining_txns = num_transactions - len(transactions)
        for _ in range(max(0, remaining_txns)):
            txn = self._create_spending_transaction(customer, month_start, month_end, device_id)
            if txn:
                transactions.append(txn)
        
        # Sort by timestamp within month
        transactions.sort(key=lambda x: x.timestamp)
        
        return transactions
    
    def _create_income_transaction(self, customer: Customer, month_start: datetime, 
                                   device_id: str) -> Optional[Transaction]:
        """Create salary/pension credit transaction."""
        try:
            txn_date = month_start.replace(day=customer.salary_day)
        except ValueError:
            txn_date = month_start.replace(day=28)
        
        txn_date = txn_date.replace(
            hour=random.randint(0, 6),  # Early morning credit
            minute=random.randint(0, 59)
        )
        
        amount = customer.monthly_income * random.uniform(0.95, 1.05)  # Small variation
        
        txn_type = TransactionType.PENSION if customer.customer_type == CustomerType.RETIRED else TransactionType.SALARY
        
        balance_before = customer.current_balance
        customer.current_balance += amount
        
        return Transaction(
            transaction_id=f"TXN{uuid.uuid4().hex[:12].upper()}",
            customer_id=customer.customer_id,
            timestamp=txn_date,
            amount=round(amount, 2),
            transaction_type=txn_type,
            merchant_name="Employer" if txn_type == TransactionType.SALARY else "Pension Office",
            merchant_category="income",
            mcc_code="MCC_0000",
            city=customer.city,
            is_credit=True,
            balance_before=round(balance_before, 2),
            balance_after=round(customer.current_balance, 2),
            status=TransactionStatus.SUCCESS,
            device_id=device_id
        )
    
    def _create_parent_transfer(self, customer: Customer, month_start: datetime, 
                                device_id: str) -> Optional[Transaction]:
        """Create parent transfer for students."""
        txn_date = month_start.replace(
            day=random.randint(1, 5),
            hour=random.randint(9, 18),
            minute=random.randint(0, 59)
        )
        
        amount = customer.monthly_income * random.uniform(0.9, 1.1)
        
        balance_before = customer.current_balance
        customer.current_balance += amount
        
        return Transaction(
            transaction_id=f"TXN{uuid.uuid4().hex[:12].upper()}",
            customer_id=customer.customer_id,
            timestamp=txn_date,
            amount=round(amount, 2),
            transaction_type=TransactionType.PARENT_TRANSFER,
            merchant_name="Parent Transfer",
            merchant_category="p2p_transfer",
            mcc_code="MCC_4829",
            city=customer.city,
            is_credit=True,
            balance_before=round(balance_before, 2),
            balance_after=round(customer.current_balance, 2),
            status=TransactionStatus.SUCCESS,
            device_id=device_id
        )
    
    def _create_business_income(self, customer: Customer, txn_date: datetime, 
                                device_id: str) -> Optional[Transaction]:
        """Create business income transaction."""
        txn_date = txn_date.replace(
            hour=random.randint(10, 18),
            minute=random.randint(0, 59)
        )
        
        # Business income is more variable
        amount = customer.monthly_income * random.uniform(0.2, 0.8)
        
        balance_before = customer.current_balance
        customer.current_balance += amount
        
        return Transaction(
            transaction_id=f"TXN{uuid.uuid4().hex[:12].upper()}",
            customer_id=customer.customer_id,
            timestamp=txn_date,
            amount=round(amount, 2),
            transaction_type=TransactionType.NEFT,
            merchant_name="Business Payment",
            merchant_category="p2p_transfer",
            mcc_code="MCC_4829",
            city=customer.city,
            is_credit=True,
            balance_before=round(balance_before, 2),
            balance_after=round(customer.current_balance, 2),
            status=TransactionStatus.SUCCESS,
            device_id=device_id
        )
    
    def _create_emi_transaction(self, customer: Customer, month_start: datetime, 
                                device_id: str) -> Optional[Transaction]:
        """Create EMI debit transaction."""
        try:
            txn_date = month_start.replace(day=customer.emi_day)
        except ValueError:
            txn_date = month_start.replace(day=28)
        
        txn_date = txn_date.replace(
            hour=random.randint(0, 8),
            minute=random.randint(0, 59)
        )
        
        amount = customer.emi_amount
        
        # Check if sufficient balance
        if customer.current_balance < amount:
            # Failed transaction due to insufficient balance
            return Transaction(
                transaction_id=f"TXN{uuid.uuid4().hex[:12].upper()}",
                customer_id=customer.customer_id,
                timestamp=txn_date,
                amount=round(amount, 2),
                transaction_type=TransactionType.EMI,
                merchant_name=random.choice(MERCHANTS["emi"])[0],
                merchant_category="emi",
                mcc_code="MCC_6012",
                city=customer.city,
                is_credit=False,
                balance_before=round(customer.current_balance, 2),
                balance_after=round(customer.current_balance, 2),  # No change
                status=TransactionStatus.FAILED,
                device_id=device_id
            )
        
        balance_before = customer.current_balance
        customer.current_balance -= amount
        
        return Transaction(
            transaction_id=f"TXN{uuid.uuid4().hex[:12].upper()}",
            customer_id=customer.customer_id,
            timestamp=txn_date,
            amount=round(amount, 2),
            transaction_type=TransactionType.EMI,
            merchant_name=random.choice(MERCHANTS["emi"])[0],
            merchant_category="emi",
            mcc_code="MCC_6012",
            city=customer.city,
            is_credit=False,
            balance_before=round(balance_before, 2),
            balance_after=round(customer.current_balance, 2),
            status=TransactionStatus.SUCCESS,
            device_id=device_id
        )
    
    def _create_rent_transaction(self, customer: Customer, month_start: datetime, 
                                 device_id: str) -> Optional[Transaction]:
        """Create rent payment transaction."""
        txn_date = month_start.replace(
            day=random.randint(1, 5),
            hour=random.randint(9, 12),
            minute=random.randint(0, 59)
        )
        
        # Rent is typically 20-40% of income
        amount = customer.monthly_income * random.uniform(0.2, 0.4)
        
        # Check if sufficient balance
        if customer.current_balance < amount:
            return None
        
        balance_before = customer.current_balance
        customer.current_balance -= amount
        
        merchant = random.choice(MERCHANTS["rent"])
        
        return Transaction(
            transaction_id=f"TXN{uuid.uuid4().hex[:12].upper()}",
            customer_id=customer.customer_id,
            timestamp=txn_date,
            amount=round(amount, 2),
            transaction_type=TransactionType.NEFT,
            merchant_name=merchant[0],
            merchant_category="rent",
            mcc_code=merchant[1],
            city=customer.city,
            is_credit=False,
            balance_before=round(balance_before, 2),
            balance_after=round(customer.current_balance, 2),
            status=TransactionStatus.SUCCESS,
            device_id=device_id
        )
    
    def _create_spending_transaction(self, customer: Customer, month_start: datetime,
                                     month_end: datetime, device_id: str) -> Optional[Transaction]:
        """Create a regular spending transaction."""
        # Random day in month
        days_in_month = (month_end - month_start).days + 1
        random_day = random.randint(0, days_in_month - 1)
        txn_date = month_start + timedelta(days=random_day)
        
        # Time based on customer's typical hours
        hour = random.randint(*customer.typical_txn_hours)
        minute = random.randint(0, 59)
        txn_date = txn_date.replace(hour=hour, minute=minute, second=random.randint(0, 59))
        
        # Weekend adjustment
        if txn_date.weekday() >= 5:  # Weekend
            if random.random() > customer.weekend_activity_multiplier:
                return None
        
        # Select category based on preferences
        category = random.choice(customer.preferred_categories)
        
        # Select merchant
        if category in MERCHANTS:
            merchant = random.choice(MERCHANTS[category])
            merchant_name, mcc_code = merchant
        else:
            merchant_name = f"Generic {category.title()}"
            mcc_code = "MCC_5999"
        
        # Determine transaction type (UPI most common)
        txn_type = random.choices(
            [TransactionType.UPI, TransactionType.DEBIT_CARD, TransactionType.CREDIT_CARD, 
             TransactionType.ATM, TransactionType.IMPS],
            weights=[0.6, 0.15, 0.1, 0.1, 0.05]
        )[0]
        
        # Amount based on category
        amount = self._get_category_amount(category, customer)
        
        # ATM withdrawals are round numbers
        if txn_type == TransactionType.ATM:
            amount = round(amount / 500) * 500
            if amount < 500:
                amount = 500
            if amount > 25000:
                amount = 25000
            merchant_name = "ATM Withdrawal"
            mcc_code = "MCC_6011"
        
        # Check if sufficient balance
        if customer.current_balance < amount:
            # Sometimes return failed transaction
            if random.random() < 0.1:  # 10% show as failed
                return Transaction(
                    transaction_id=f"TXN{uuid.uuid4().hex[:12].upper()}",
                    customer_id=customer.customer_id,
                    timestamp=txn_date,
                    amount=round(amount, 2),
                    transaction_type=txn_type,
                    merchant_name=merchant_name,
                    merchant_category=category,
                    mcc_code=mcc_code,
                    city=customer.city,
                    is_credit=False,
                    balance_before=round(customer.current_balance, 2),
                    balance_after=round(customer.current_balance, 2),
                    status=TransactionStatus.FAILED,
                    device_id=device_id
                )
            return None
        
        # Check for refund (5% chance on ecommerce)
        is_refund = False
        if category == "ecommerce" and random.random() < 0.05:
            is_refund = True
            amount = amount * random.uniform(0.3, 1.0)  # Partial or full refund
        
        balance_before = customer.current_balance
        if is_refund:
            customer.current_balance += amount
            is_credit = True
            txn_type = TransactionType.REFUND
        else:
            customer.current_balance -= amount
            is_credit = False
        
        return Transaction(
            transaction_id=f"TXN{uuid.uuid4().hex[:12].upper()}",
            customer_id=customer.customer_id,
            timestamp=txn_date,
            amount=round(amount, 2),
            transaction_type=txn_type,
            merchant_name=merchant_name,
            merchant_category=category,
            mcc_code=mcc_code,
            city=customer.city,
            is_credit=is_credit,
            balance_before=round(balance_before, 2),
            balance_after=round(customer.current_balance, 2),
            status=TransactionStatus.SUCCESS,
            device_id=device_id
        )
    
    def _get_category_amount(self, category: str, customer: Customer) -> float:
        """Get typical transaction amount for a category."""
        # Base amounts by category (in INR)
        category_amounts = {
            "grocery": (200, 5000),
            "food_delivery": (150, 1500),
            "ecommerce": (300, 15000),
            "fuel": (500, 5000),
            "utilities": (200, 5000),
            "travel": (100, 20000),
            "entertainment": (100, 2000),
            "healthcare": (100, 10000),
            "education": (500, 50000),
            "investment": (1000, 50000),
            "p2p_transfer": (500, 50000),
        }
        
        min_amt, max_amt = category_amounts.get(category, (100, 5000))
        
        # Adjust based on customer income
        income_multiplier = min(2.0, max(0.5, customer.monthly_income / 50000))
        max_amt = min(max_amt * income_multiplier, customer.current_balance * 0.5)
        
        if min_amt > max_amt:
            min_amt = max_amt * 0.5
        
        # Log-normal distribution for more realistic amounts (more small transactions)
        mean = np.log((min_amt + max_amt) / 3)
        sigma = 0.8
        amount = np.random.lognormal(mean, sigma)
        
        return max(min_amt, min(max_amt, amount))
    
    def _inject_fraud_patterns(self, transactions: list[Transaction], 
                               customers: list[Customer]) -> list[Transaction]:
        """Inject fraud patterns into transactions."""
        # Calculate target fraud count
        target_fraud_count = int(len(transactions) * self.fraud_probability)
        
        print(f"Injecting {target_fraud_count} fraud transactions...")
        
        # Create customer lookup
        customer_lookup = {c.customer_id: c for c in customers}
        
        # Select random transactions to convert to fraud
        fraud_indices = random.sample(range(len(transactions)), min(target_fraud_count, len(transactions)))
        
        fraud_types = [
            ("account_takeover", 0.4),
            ("card_cloning", 0.3),
            ("unusual_merchant", 0.2),
            ("velocity_attack", 0.1)
        ]
        
        for idx in fraud_indices:
            txn = transactions[idx]
            
            # Skip credit transactions and already fraud
            if txn.is_credit or txn.is_fraud:
                continue
            
            # Select fraud type
            fraud_type = random.choices(
                [ft[0] for ft in fraud_types],
                weights=[ft[1] for ft in fraud_types]
            )[0]
            
            # Modify transaction to look fraudulent
            txn = self._make_fraudulent(txn, fraud_type, customer_lookup.get(txn.customer_id))
            transactions[idx] = txn
        
        return transactions
    
    def _make_fraudulent(self, txn: Transaction, fraud_type: str, 
                         customer: Optional[Customer]) -> Transaction:
        """Modify a transaction to have fraud characteristics."""
        txn.is_fraud = True
        txn.fraud_type = fraud_type
        
        # Calculate max amount that won't cause negative balance
        max_fraud_amount = txn.balance_before * 0.95  # Leave 5% buffer
        
        if fraud_type == "account_takeover":
            # Large amount, unusual time, different city
            desired_amount = txn.amount * random.uniform(3, 10)
            txn.amount = min(desired_amount, max_fraud_amount)
            txn.timestamp = txn.timestamp.replace(hour=random.randint(1, 5))  # Late night
            txn.city = random.choice(CITIES["tier1"] + CITIES["tier2"])
            txn.device_id = f"DEV{uuid.uuid4().hex[:8].upper()}"  # New device
            
        elif fraud_type == "card_cloning":
            # Transaction from different city within short time
            txn.city = random.choice([c for c in CITIES["tier1"] + CITIES["tier2"] 
                                      if c != txn.city])
            desired_amount = txn.amount * random.uniform(1.5, 5)
            txn.amount = min(desired_amount, max_fraud_amount)
            
        elif fraud_type == "unusual_merchant":
            # First-time high-value at risky category
            risky_merchants = [
                ("Crypto Exchange", "MCC_6051"), ("Foreign Wire", "MCC_4829"),
                ("Casino Online", "MCC_7995"), ("Adult Content", "MCC_5967")
            ]
            merchant = random.choice(risky_merchants)
            txn.merchant_name = merchant[0]
            txn.mcc_code = merchant[1]
            desired_amount = txn.amount * random.uniform(5, 20)
            txn.amount = min(desired_amount, max_fraud_amount)
            
        elif fraud_type == "velocity_attack":
            # Multiple small transactions (we mark this one, pattern in features)
            txn.amount = random.uniform(100, 500)  # Small amount
        
        # Recalculate balance (ensure non-negative)
        txn.balance_after = max(0, txn.balance_before - txn.amount)
        
        return txn


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def generate_dataset(seed: int = 42, output_dir: str = "data") -> tuple[pd.DataFrame, pd.DataFrame]:
    """Generate the complete dataset."""
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    print("=" * 60)
    print("Indian Bank Transaction Data Generator")
    print("=" * 60)
    
    # Generate customers
    print("\n[1/3] Generating customer profiles...")
    customer_gen = CustomerGenerator(seed=seed)
    customers = customer_gen.generate_customers()
    print(f"Created {len(customers)} customers")
    
    # Generate transactions
    print("\n[2/3] Generating transaction history...")
    txn_gen = TransactionGenerator(seed=seed)
    transactions = txn_gen.generate_transactions(customers)
    print(f"Created {len(transactions)} transactions")
    
    # Convert to DataFrames
    print("\n[3/3] Saving to parquet files...")
    
    # Customers DataFrame
    customers_df = pd.DataFrame([
        {
            "customer_id": c.customer_id,
            "customer_type": c.customer_type.value,
            "name": c.name,
            "age": c.age,
            "city": c.city,
            "city_tier": c.city_tier,
            "monthly_income": c.monthly_income,
            "initial_balance": c.initial_balance,
            "final_balance": c.current_balance,
            "account_created": c.account_created,
            "has_emi": c.has_emi,
            "emi_amount": c.emi_amount,
            "salary_day": c.salary_day,
            "avg_monthly_transactions": c.avg_monthly_transactions
        }
        for c in customers
    ])
    
    # Transactions DataFrame
    transactions_df = pd.DataFrame([
        {
            "transaction_id": t.transaction_id,
            "customer_id": t.customer_id,
            "timestamp": t.timestamp,
            "amount": t.amount,
            "transaction_type": t.transaction_type.value,
            "merchant_name": t.merchant_name,
            "merchant_category": t.merchant_category,
            "mcc_code": t.mcc_code,
            "city": t.city,
            "is_credit": t.is_credit,
            "balance_before": t.balance_before,
            "balance_after": t.balance_after,
            "status": t.status.value,
            "device_id": t.device_id,
            "is_fraud": t.is_fraud,
            "fraud_type": t.fraud_type
        }
        for t in transactions
    ])
    
    # Save to parquet
    customers_path = os.path.join(output_dir, "customers.parquet")
    transactions_path = os.path.join(output_dir, "transactions.parquet")
    
    customers_df.to_parquet(customers_path, index=False)
    transactions_df.to_parquet(transactions_path, index=False)
    
    # Print summary
    print("\n" + "=" * 60)
    print("GENERATION COMPLETE")
    print("=" * 60)
    print(f"\nCustomers: {len(customers_df)}")
    print(f"  - Salaried: {len(customers_df[customers_df['customer_type'] == 'salaried'])}")
    print(f"  - Self-Employed: {len(customers_df[customers_df['customer_type'] == 'self_employed'])}")
    print(f"  - Students: {len(customers_df[customers_df['customer_type'] == 'student'])}")
    print(f"  - Small Business: {len(customers_df[customers_df['customer_type'] == 'small_business'])}")
    print(f"  - Retired: {len(customers_df[customers_df['customer_type'] == 'retired'])}")
    
    print(f"\nTransactions: {len(transactions_df)}")
    print(f"  - Successful: {len(transactions_df[transactions_df['status'] == 'SUCCESS'])}")
    print(f"  - Failed: {len(transactions_df[transactions_df['status'] == 'FAILED'])}")
    print(f"  - Fraud: {len(transactions_df[transactions_df['is_fraud'] == True])} ({len(transactions_df[transactions_df['is_fraud'] == True])/len(transactions_df)*100:.3f}%)")
    
    print(f"\nBalance Check:")
    print(f"  - Min balance after: ₹{transactions_df['balance_after'].min():,.2f}")
    print(f"  - Negative balances: {len(transactions_df[transactions_df['balance_after'] < 0])}")
    
    print(f"\nFiles saved:")
    print(f"  - {customers_path}")
    print(f"  - {transactions_path}")
    
    return customers_df, transactions_df


if __name__ == "__main__":
    customers_df, transactions_df = generate_dataset()

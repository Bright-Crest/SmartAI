# 区块链交互模块
"""
这个模块包含与区块链交互的工具和功能
"""

from app.blockchain.ethereum import (
    get_web3,
    is_mock_mode,
    get_balance,
    get_transaction_receipt,
    get_token_balances,
    get_address_transactions
)

from app.blockchain.contract import (
    get_contract_instance,
    contract_call,
    get_events
)

__all__ = [
    'get_web3',
    'is_mock_mode',
    'get_balance',
    'get_transaction_receipt',
    'get_token_balances',
    'get_address_transactions',
    'get_contract_instance',
    'contract_call',
    'get_events'
]

"""
区块链模块
提供以太坊区块链交互功能
""" 
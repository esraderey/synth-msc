import json
import time
import ecdsa

class ShyntToken:
    def __init__(self, name="Shynt Token", symbol="SHYNT", initial_supply=0):
        self.name = name
        self.symbol = symbol
        self.total_supply = initial_supply
        # Balances: {address: balance}
        self.balances = {}

    def create_wallet_balance(self, address):
        if address not in self.balances:
            self.balances[address] = 0

    def mint(self, address, amount):
        self.create_wallet_balance(address)
        self.balances[address] += amount
        self.total_supply += amount
        print(f"Minted {amount} {self.symbol} to {address}")

    def transfer(self, sender, recipient, amount, signature, public_key_hex, message):
        """
        Transfiere tokens del emisor (sender) al receptor, validando la firma.
        El mensaje debe ser un JSON string que incluya sender, recipient, amount y timestamp.
        """
        try:
            vk = ecdsa.VerifyingKey.from_string(bytes.fromhex(public_key_hex), curve=ecdsa.SECP256k1)
            if not vk.verify(bytes.fromhex(signature), message.encode("utf-8")):
                return False, "Firma inválida"
        except Exception as e:
            return False, f"Error en verificación de firma: {e}"
        
        self.create_wallet_balance(sender)
        self.create_wallet_balance(recipient)
        if self.balances[sender] < amount:
            return False, "Saldo insuficiente"
        
        self.balances[sender] -= amount
        self.balances[recipient] += amount
        return True, "Transferencia exitosa"

    def get_balance(self, address):
        self.create_wallet_balance(address)
        return self.balances[address]

    def to_json(self):
        return json.dumps({
            "name": self.name,
            "symbol": self.symbol,
            "total_supply": self.total_supply,
            "balances": self.balances
        }, indent=2)


# Ejemplo de uso:
if __name__ == "__main__":
    from wallet import Wallet  # Asegúrate de tener wallet.py correctamente configurado

    # Crear dos wallets
    wallet1 = Wallet()
    wallet2 = Wallet()

    addr1 = wallet1.get_address()
    addr2 = wallet2.get_address()

    token = ShyntToken(initial_supply=0)

    # Mint 100 SHYNT a wallet1
    token.mint(addr1, 100)
    print("Saldo wallet1:", token.get_balance(addr1))
    print("Saldo wallet2:", token.get_balance(addr2))

    # Preparar transferencia de wallet1 a wallet2 de 30 SHYNT
    tx_data = {
        "sender": addr1,
        "recipient": addr2,
        "amount": 30,
        "timestamp": time.time()
    }
    message = json.dumps(tx_data, sort_keys=True)
    signature = wallet1.sign(message)
    sender_pub = wallet1.get_public_key()

    success, msg = token.transfer(addr1, addr2, 30, signature, sender_pub, message)
    print(msg)
    print("Saldo wallet1 después:", token.get_balance(addr1))
    print("Saldo wallet2 después:", token.get_balance(addr2))
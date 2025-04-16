import ecdsa
import hashlib
import base58
import time

class Wallet:
    def __init__(self):
        # Genera una clave privada usando la curva SECP256k1
        self.private_key = ecdsa.SigningKey.generate(curve=ecdsa.SECP256k1)
        self.public_key = self.private_key.get_verifying_key()

    def sign(self, message):
        # Firma el mensaje (string) y retorna la firma en hexadecimal.
        message_bytes = message.encode('utf-8')
        signature = self.private_key.sign(message_bytes)
        return signature.hex()
        
    def verify(self, message, signature):
        # Verifica la firma (hexadecimal) para el mensaje dado.
        try:
            message_bytes = message.encode('utf-8')
            return self.public_key.verify(bytes.fromhex(signature), message_bytes)
        except Exception:
            return False

    def get_public_key(self):
        # Devuelve la clave pública en hexadecimal.
        return self.public_key.to_string().hex()

    def get_address(self):
        # Genera una dirección basada en la clave pública.
        pub_key_bytes = self.public_key.to_string()
        sha = hashlib.sha256(pub_key_bytes).digest()
        ripemd160 = hashlib.new('ripemd160', sha).digest()
        address = base58.b58encode(ripemd160)
        return address.decode('utf-8')

if __name__ == "__main__":
    wallet = Wallet()
    print("Wallet creada:")
    print(f"Clave Pública: {wallet.get_public_key()}")
    print(f"Dirección: {wallet.get_address()}")
    message = "Mensaje de prueba"
    signature = wallet.sign(message)
    print(f"Firma del mensaje '{message}': {signature}")
    verify_result = wallet.verify(message, signature)
    print(f"Verificación de la firma: {verify_result}")
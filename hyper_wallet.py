import ecdsa
import hashlib
import base58
import time
import json
import uuid
import os
import secrets
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from typing import Dict, List, Any, Optional, Tuple, Union

class AssetAccount:
    """Cuenta para un tipo específico de activo digital"""
    
    def __init__(self, asset_type: str, initial_balance: float = 0.0):
        self.asset_type = asset_type
        self.balance = initial_balance
        self.history = []
        self.metadata = {}
        self.created_at = time.time()
    
    def update_balance(self, amount: float, tx_reference: str, description: str) -> bool:
        """Actualiza el balance con registro histórico"""
        prev_balance = self.balance
        self.balance += amount
        
        # Si es un gasto, asegurar que hay fondos suficientes
        if amount < 0 and self.balance < 0:
            self.balance = prev_balance  # Revertir
            return False
            
        # Registrar la transacción en el historial
        self.history.append({
            "timestamp": time.time(),
            "amount": amount,
            "previous_balance": prev_balance,
            "new_balance": self.balance,
            "reference": tx_reference,
            "description": description
        })
        
        return True
    
    def get_statement(self, start_time: Optional[float] = None, end_time: Optional[float] = None) -> List[Dict]:
        """Obtiene extracto de transacciones filtrado por fechas"""
        if start_time is None and end_time is None:
            return self.history
            
        filtered = []
        for tx in self.history:
            tx_time = tx["timestamp"]
            if start_time and tx_time < start_time:
                continue
            if end_time and tx_time > end_time:
                continue
            filtered.append(tx)
            
        return filtered
    
    def to_dict(self) -> Dict:
        """Serializa la cuenta a diccionario"""
        return {
            "asset_type": self.asset_type,
            "balance": self.balance,
            "history": self.history,
            "metadata": self.metadata,
            "created_at": self.created_at
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'AssetAccount':
        """Crea una cuenta desde diccionario"""
        account = cls(data["asset_type"], data["balance"])
        account.history = data["history"]
        account.metadata = data["metadata"]
        account.created_at = data["created_at"]
        return account
        
    def __repr__(self) -> str:
        return f"AssetAccount({self.asset_type}: {self.balance})"


class VerifiableCredential:
    """Implementación básica de una credencial verificable"""
    
    def __init__(self, 
                 subject: str, 
                 issuer: str, 
                 claim_type: str,
                 claims: Dict[str, Any],
                 expiration: Optional[float] = None):
        """
        Args:
            subject: El DID del sujeto de la credencial
            issuer: El DID del emisor de la credencial
            claim_type: Tipo de credencial (Ej: "KnowledgeDomainExpert")
            claims: Contenido de la credencial
            expiration: Tiempo de expiración (None = no expira)
        """
        self.id = str(uuid.uuid4())
        self.subject = subject
        self.issuer = issuer
        self.claim_type = claim_type
        self.claims = claims
        self.issued_at = time.time()
        self.expiration = expiration
        self.proof = None  # Se establece al firmar
    
    def sign(self, signing_key: ecdsa.SigningKey) -> None:
        """Firma la credencial y establece la prueba"""
        content = self._get_signable_representation()
        signature = signing_key.sign(content.encode('utf-8'))
        
        self.proof = {
            "type": "EcdsaSecp256k1Signature2019",
            "created": time.time(),
            "verificationMethod": f"{self.issuer}#keys-1",
            "signature": signature.hex()
        }
    
    def verify(self, verifying_key: ecdsa.VerifyingKey) -> bool:
        """Verifica la firma de la credencial"""
        if not self.proof:
            return False
            
        try:
            content = self._get_signable_representation()
            signature_bytes = bytes.fromhex(self.proof["signature"])
            return verifying_key.verify(signature_bytes, content.encode('utf-8'))
        except Exception:
            return False
    
    def is_expired(self) -> bool:
        """Comprueba si la credencial ha expirado"""
        if not self.expiration:
            return False
        return time.time() > self.expiration
    
    def _get_signable_representation(self) -> str:
        """Obtiene una representación canonicalizada para firmar"""
        # Excluir la prueba (proof) al generar la representación para firmar
        data = {
            "id": self.id,
            "subject": self.subject,
            "issuer": self.issuer,
            "claim_type": self.claim_type,
            "claims": self.claims,
            "issued_at": self.issued_at,
            "expiration": self.expiration
        }
        return json.dumps(data, sort_keys=True)
    
    def to_dict(self) -> Dict:
        """Serializa la credencial a diccionario"""
        return {
            "id": self.id,
            "subject": self.subject,
            "issuer": self.issuer,
            "claim_type": self.claim_type,
            "claims": self.claims,
            "issued_at": self.issued_at,
            "expiration": self.expiration,
            "proof": self.proof
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'VerifiableCredential':
        """Crea una credencial desde diccionario"""
        credential = cls(
            data["subject"],
            data["issuer"],
            data["claim_type"],
            data["claims"],
            data["expiration"]
        )
        credential.id = data["id"]
        credential.issued_at = data["issued_at"]
        credential.proof = data["proof"]
        return credential


class KeyManager:
    """Gestor avanzado de claves criptográficas"""
    
    def __init__(self):
        self.keys = {}
        self.default_key = None
        self.key_derivations = {}  # Para claves derivadas jerárquicamente
    
    def generate_key(self, key_id: str = None, curve: ecdsa.curves.Curve = ecdsa.SECP256k1) -> str:
        """Genera una nueva clave y la almacena"""
        if not key_id:
            key_id = str(uuid.uuid4())
            
        if key_id in self.keys:
            raise ValueError(f"Key ID {key_id} already exists")
            
        sk = ecdsa.SigningKey.generate(curve=curve)
        vk = sk.get_verifying_key()
        
        self.keys[key_id] = {
            "private_key": sk,
            "public_key": vk,
            "curve": curve,
            "created_at": time.time(),
            "metadata": {}
        }
        
        # Si es la primera clave, establecerla como default
        if not self.default_key:
            self.default_key = key_id
            
        return key_id
    
    def derive_key(self, parent_key_id: str, derivation_path: str) -> str:
        """Deriva una clave jerárquicamente desde una clave padre"""
        if parent_key_id not in self.keys:
            raise ValueError(f"Parent key {parent_key_id} not found")
        
        # Usar HMAC para derivar una nueva clave a partir de la clave privada padre
        parent_key = self.keys[parent_key_id]["private_key"].to_string()
        derived_seed = hashlib.pbkdf2_hmac(
            'sha256', 
            parent_key, 
            derivation_path.encode(), 
            iterations=50000, 
            dklen=32
        )
        
        # Generar la nueva clave a partir de la semilla derivada
        derived_key_id = f"{parent_key_id}/{derivation_path}"
        derived_sk = ecdsa.SigningKey.from_string(derived_seed, curve=self.keys[parent_key_id]["curve"])
        derived_vk = derived_sk.get_verifying_key()
        
        self.keys[derived_key_id] = {
            "private_key": derived_sk,
            "public_key": derived_vk,
            "curve": self.keys[parent_key_id]["curve"],
            "created_at": time.time(),
            "metadata": {"derived_from": parent_key_id, "path": derivation_path}
        }
        
        # Registrar la derivación
        if parent_key_id not in self.key_derivations:
            self.key_derivations[parent_key_id] = []
        self.key_derivations[parent_key_id].append(derived_key_id)
        
        return derived_key_id
    
    def get_key(self, key_id: Optional[str] = None) -> Dict:
        """Obtiene información sobre una clave"""
        if not key_id:
            key_id = self.default_key
            
        if key_id not in self.keys:
            raise ValueError(f"Key {key_id} not found")
            
        return self.keys[key_id]
    
    def get_public_key_hex(self, key_id: Optional[str] = None) -> str:
        """Obtiene la clave pública en formato hexadecimal"""
        key_data = self.get_key(key_id)
        return key_data["public_key"].to_string().hex()
    
    def sign(self, message: str, key_id: Optional[str] = None) -> str:
        """Firma un mensaje con la clave especificada"""
        key_data = self.get_key(key_id)
        message_bytes = message.encode('utf-8')
        signature = key_data["private_key"].sign(message_bytes)
        return signature.hex()
    
    def verify(self, message: str, signature: str, key_id: Optional[str] = None) -> bool:
        """Verifica una firma con la clave especificada"""
        try:
            key_data = self.get_key(key_id)
            message_bytes = message.encode('utf-8')
            signature_bytes = bytes.fromhex(signature)
            return key_data["public_key"].verify(signature_bytes, message_bytes)
        except Exception:
            return False
    
    def export_key_info(self, include_private: bool = False) -> Dict:
        """Exporta información sobre las claves (sin claves privadas por defecto)"""
        result = {}
        for key_id, key_data in self.keys.items():
            key_info = {
                "public_key": key_data["public_key"].to_string().hex(),
                "curve": key_data["curve"].name,
                "created_at": key_data["created_at"],
                "metadata": key_data["metadata"]
            }
            
            if include_private:
                key_info["private_key"] = key_data["private_key"].to_string().hex()
                
            result[key_id] = key_info
            
        return result
    
    def encrypt_to_self(self, plaintext: str, key_id: Optional[str] = None) -> bytes:
        """Encripta un mensaje para uno mismo usando AESGCM"""
        # Generar una clave AES derivada de la clave privada
        key_data = self.get_key(key_id)
        private_key_bytes = key_data["private_key"].to_string()
        
        # Generar salt y nonce
        salt = os.urandom(16)
        nonce = os.urandom(12)
        
        # Derivar clave AES
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=100000,
        )
        aes_key = kdf.derive(private_key_bytes)
        
        # Encriptar
        aesgcm = AESGCM(aes_key)
        ciphertext = aesgcm.encrypt(nonce, plaintext.encode('utf-8'), None)
        
        # Combinar salt, nonce y ciphertext
        return salt + nonce + ciphertext
    
    def decrypt_from_self(self, ciphertext: bytes, key_id: Optional[str] = None) -> str:
        """Desencripta un mensaje encriptado para uno mismo"""
        # Extraer salt y nonce
        salt = ciphertext[:16]
        nonce = ciphertext[16:28]
        actual_ciphertext = ciphertext[28:]
        
        # Derivar clave AES
        key_data = self.get_key(key_id)
        private_key_bytes = key_data["private_key"].to_string()
        
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=100000,
        )
        aes_key = kdf.derive(private_key_bytes)
        
        # Desencriptar
        aesgcm = AESGCM(aes_key)
        plaintext = aesgcm.decrypt(nonce, actual_ciphertext, None)
        
        return plaintext.decode('utf-8')


class HyperWallet:
    """Billetera digital avanzada con múltiples capacidades"""
    
    # Constantes para tipos de activos
    MSC_TOKEN = "$MSC"    # Token para gobernanza
    OMEGA_TOKEN = "Ω"     # Unidad de energía computacional
    PHI_TOKEN = "Φ"       # Métrica de impacto (minada por consenso epistémico)
    PSI_TOKEN = "Ψ"       # Token de reputación epistémica
    
    def __init__(self, wallet_id: Optional[str] = None):
        """Initialize a new HyperWallet or load an existing one"""
        self.wallet_id = wallet_id or str(uuid.uuid4())
        self.created_at = time.time()
        self.key_manager = KeyManager()
        
        # Generar clave por defecto
        self.default_key_id = self.key_manager.generate_key("default")
        
        # Inicializar cuentas para distintos tipos de tokens
        self.accounts = {
            self.MSC_TOKEN: AssetAccount(self.MSC_TOKEN),
            self.OMEGA_TOKEN: AssetAccount(self.OMEGA_TOKEN),
            self.PHI_TOKEN: AssetAccount(self.PHI_TOKEN),
            self.PSI_TOKEN: AssetAccount(self.PSI_TOKEN)
        }
        
        # Credenciales verificables recibidas
        self.credentials = []
        
        # Contactos y canales de comunicación seguros
        self.contacts = {}
        self.secure_channels = {}
        
        # Metadatos de la wallet
        self.metadata = {}
        
        # ID descentralizado (DID)
        self.did = self._generate_did()
        
        # Historial de actividad general
        self.activity_log = []
    
    def _generate_did(self) -> str:
        """Genera un Identificador Descentralizado basado en la clave pública principal"""
        pub_key = self.key_manager.get_public_key_hex()
        # Formato simple: did:hyper:{hash-derivado-de-clave-pública}
        key_hash = hashlib.sha256(bytes.fromhex(pub_key)).hexdigest()[:16]
        return f"did:hyper:{key_hash}"
    
    def get_address(self, key_id: Optional[str] = None) -> str:
        """Genera una dirección compatible con blockchain basada en la clave especificada"""
        key_info = self.key_manager.get_key(key_id)
        pub_key_bytes = key_info["public_key"].to_string()
        sha = hashlib.sha256(pub_key_bytes).digest()
        ripemd160 = hashlib.new('ripemd160', sha).digest()
        
        # Añadir versión para mejor compatibilidad
        versioned = b'\x00' + ripemd160
        
        # Añadir checksum
        checksum = hashlib.sha256(hashlib.sha256(versioned).digest()).digest()[:4]
        address_bytes = versioned + checksum
        
        # Codificar en base58
        address = base58.b58encode(address_bytes)
        return address.decode('utf-8')
    
    def create_asset_account(self, asset_type: str, initial_balance: float = 0.0) -> AssetAccount:
        """Crea una nueva cuenta para un tipo de activo específico"""
        if asset_type in self.accounts:
            raise ValueError(f"Account for {asset_type} already exists")
            
        account = AssetAccount(asset_type, initial_balance)
        self.accounts[asset_type] = account
        
        self._log_activity("create_account", {
            "asset_type": asset_type,
            "initial_balance": initial_balance
        })
        
        return account
    
    def get_account(self, asset_type: str) -> AssetAccount:
        """Obtiene la cuenta para un tipo de activo específico"""
        if asset_type not in self.accounts:
            raise ValueError(f"No account found for asset type {asset_type}")
        return self.accounts[asset_type]
    
    def transfer(self, asset_type: str, amount: float, recipient_address: str, description: str = "") -> str:
        """
        Prepara una transferencia de activos (para ser ejecutada por una blockchain)
        Retorna la firma de la transacción
        """
        if asset_type not in self.accounts:
            raise ValueError(f"No account found for asset type {asset_type}")
            
        account = self.accounts[asset_type]
        if account.balance < amount:
            raise ValueError(f"Insufficient balance of {asset_type}")
            
        # Crear objeto de transacción
        tx_id = str(uuid.uuid4())
        tx = {
            "id": tx_id,
            "sender": self.get_address(),
            "recipient": recipient_address,
            "asset_type": asset_type,
            "amount": amount,
            "description": description,
            "timestamp": time.time()
        }
        
        # Firmar la transacción
        tx_json = json.dumps(tx, sort_keys=True)
        signature = self.key_manager.sign(tx_json)
        
        # Registrar en el historial local (pendiente de confirmación en blockchain)
        account.update_balance(-amount, tx_id, f"Transfer to {recipient_address}: {description}")
        
        self._log_activity("transfer_out", {
            "tx_id": tx_id,
            "asset_type": asset_type,
            "amount": amount,
            "recipient": recipient_address
        })
        
        return signature
    
    def receive(self, asset_type: str, amount: float, sender_address: str, tx_id: str, description: str = "") -> bool:
        """Registra recepción de fondos (tras verificación en blockchain)"""
        if asset_type not in self.accounts:
            self.create_asset_account(asset_type)
            
        account = self.accounts[asset_type]
        result = account.update_balance(amount, tx_id, f"Received from {sender_address}: {description}")
        
        if result:
            self._log_activity("transfer_in", {
                "tx_id": tx_id,
                "asset_type": asset_type,
                "amount": amount,
                "sender": sender_address
            })
            
        return result
    
    def issue_credential(self, subject_did: str, claim_type: str, claims: Dict) -> VerifiableCredential:
        """
        Emite una credencial verificable para un sujeto
        """
        credential = VerifiableCredential(
            subject=subject_did,
            issuer=self.did,
            claim_type=claim_type,
            claims=claims,
            expiration=time.time() + 31536000  # Default: 1 año
        )
        
        # Firmar la credencial con la clave por defecto
        credential.sign(self.key_manager.get_key()["private_key"])
        
        self._log_activity("issue_credential", {
            "credential_id": credential.id,
            "subject": subject_did,
            "claim_type": claim_type
        })
        
        return credential
    
    def add_credential(self, credential: Union[VerifiableCredential, Dict]) -> bool:
        """
        Añade una credencial recibida a la wallet
        """
        if isinstance(credential, dict):
            credential = VerifiableCredential.from_dict(credential)
            
        # Verificar que la credencial es para este wallet
        if credential.subject != self.did:
            raise ValueError(f"Credential subject ({credential.subject}) does not match wallet DID ({self.did})")
        
        # Verificar que no está expirada
        if credential.is_expired():
            return False
            
        # Añadir a las credenciales
        self.credentials.append(credential)
        
        self._log_activity("add_credential", {
            "credential_id": credential.id,
            "issuer": credential.issuer,
            "claim_type": credential.claim_type
        })
        
        return True
    
    def get_credentials(self, claim_type: Optional[str] = None) -> List[VerifiableCredential]:
        """
        Obtiene las credenciales, opcionalmente filtradas por tipo
        """
        if not claim_type:
            return self.credentials
            
        return [c for c in self.credentials if c.claim_type == claim_type]
    
    def add_contact(self, 
                    did: str, 
                    name: str, 
                    public_key: str, 
                    metadata: Optional[Dict] = None) -> str:
        """
        Añade un contacto a la wallet
        """
        contact_id = str(uuid.uuid4())
        self.contacts[contact_id] = {
            "did": did,
            "name": name,
            "public_key": public_key,
            "added_at": time.time(),
            "metadata": metadata or {}
        }
        
        self._log_activity("add_contact", {
            "contact_id": contact_id,
            "did": did,
            "name": name
        })
        
        return contact_id
    
    def establish_secure_channel(self, contact_id: str) -> str:
        """
        Establece un canal seguro con un contacto
        """
        if contact_id not in self.contacts:
            raise ValueError(f"Contact {contact_id} not found")
            
        # Generar una clave compartida (simulada)
        channel_id = str(uuid.uuid4())
        shared_secret = secrets.token_hex(32)
        
        self.secure_channels[channel_id] = {
            "contact_id": contact_id,
            "established_at": time.time(),
            "shared_secret": shared_secret,
            "status": "active"
        }
        
        self._log_activity("establish_channel", {
            "channel_id": channel_id,
            "contact_id": contact_id,
            "contact_name": self.contacts[contact_id]["name"]
        })
        
        return channel_id
    
    def encrypt_message(self, channel_id: str, message: str) -> bytes:
        """
        Encripta un mensaje usando un canal seguro
        """
        if channel_id not in self.secure_channels:
            raise ValueError(f"Secure channel {channel_id} not found")
            
        channel = self.secure_channels[channel_id]
        if channel["status"] != "active":
            raise ValueError(f"Channel {channel_id} is not active")
            
        # Usar el secreto compartido para derivar una clave para esta mensaje específico
        salt = os.urandom(16)
        nonce = os.urandom(12)
        
        # En una implementación real, ambas partes utilizarían
        # un algoritmo de acuerdo de claves como ECDH
        shared_secret = bytes.fromhex(channel["shared_secret"])
        
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=100000,
        )
        message_key = kdf.derive(shared_secret)
        
        # Encriptar
        aesgcm = AESGCM(message_key)
        ciphertext = aesgcm.encrypt(nonce, message.encode('utf-8'), None)
        
        # Combinar salt, nonce y ciphertext
        encrypted_message = salt + nonce + ciphertext
        
        self._log_activity("encrypt_message", {
            "channel_id": channel_id,
            "contact_id": channel["contact_id"],
            "timestamp": time.time()
        })
        
        return encrypted_message
    
    def _log_activity(self, activity_type: str, details: Dict) -> None:
        """
        Registra actividad en el log de la wallet
        """
        entry = {
            "timestamp": time.time(),
            "type": activity_type,
            "details": details
        }
        self.activity_log.append(entry)
    
    def export_state(self, include_private: bool = False) -> Dict:
        """
        Exporta el estado completo de la wallet para respaldo o sincronización
        """
        state = {
            "wallet_id": self.wallet_id,
            "created_at": self.created_at,
            "did": self.did,
            "accounts": {name: account.to_dict() for name, account in self.accounts.items()},
            "credentials": [cred.to_dict() for cred in self.credentials],
            "contacts": self.contacts,
            "metadata": self.metadata,
            "activity_log": self.activity_log
        }
        
        # La exportación de claves privadas se controla con un flag por seguridad
        if include_private:
            state["keys"] = self.key_manager.export_key_info(include_private=True)
            state["secure_channels"] = self.secure_channels
        else:
            state["keys"] = self.key_manager.export_key_info(include_private=False)
        
        return state
    
    @classmethod
    def from_state(cls, state: Dict) -> 'HyperWallet':
        """
        Crea una wallet a partir de un estado exportado
        """
        # Esta es una implementación simplificada;
        # en una versión real se necesitaría manejar la importación de claves privadas
        wallet = cls(state["wallet_id"])
        wallet.created_at = state["created_at"]
        wallet.did = state["did"]
        
        # Restaurar cuentas
        wallet.accounts = {
            name: AssetAccount.from_dict(account_data) 
            for name, account_data in state["accounts"].items()
        }
        
        # Restaurar credenciales
        wallet.credentials = [
            VerifiableCredential.from_dict(cred_data)
            for cred_data in state["credentials"]
        ]
        
        wallet.contacts = state["contacts"]
        wallet.metadata = state["metadata"]
        wallet.activity_log = state["activity_log"]
        
        # Nota: Esta implementación no restaura las claves privadas ni los canales seguros
        # Se necesitaría una implementación más compleja para manejar eso de forma segura
        
        return wallet
        
    def __repr__(self) -> str:
        """Representación de string para la wallet"""
        account_str = ", ".join([f"{name}: {acct.balance}" for name, acct in self.accounts.items()])
        return f"HyperWallet(DID={self.did}, Accounts=[{account_str}])"


# --- Example usage ---
if __name__ == "__main__":
    # Create a new HyperWallet
    wallet = HyperWallet()
    print("HyperWallet created:")
    print(f"DID: {wallet.did}")
    print(f"Blockchain Address: {wallet.get_address()}")
    
    # Add some tokens
    wallet.receive(HyperWallet.MSC_TOKEN, 100.0, "genesis", "initial_allocation", "Initial token allocation")
    wallet.receive(HyperWallet.PHI_TOKEN, 25.0, "consensus_reward", "reward_tx_1", "Reward for knowledge contribution")
    
    # Create and verify a credential
    expert_credential = wallet.issue_credential(
        "did:hyper:example",
        "DomainExpert",
        {"domain": "blockchain", "level": "advanced", "verified": True}
    )
    print(f"\nIssued credential: {expert_credential.id}")
    
    # Encrypt and decrypt message to self
    key_manager = wallet.key_manager
    message = "Confidential wallet data"
    encrypted = key_manager.encrypt_to_self(message)
    decrypted = key_manager.decrypt_from_self(encrypted)
    print(f"\nEncryption test: {message == decrypted}")
    
    # Export wallet state (without private keys)
    state = wallet.export_state()
    print(f"\nWallet state (account balances):")
    for name, account in wallet.accounts.items():
        print(f"  {name}: {account.balance}")
    
    # Demonstrate hierarchical key derivation
    child_key_id = key_manager.derive_key("default", "m/0/1")
    child_address = wallet.get_address(child_key_id)
    print(f"\nDerived child key: {child_key_id}")
    print(f"Child address: {child_address}")
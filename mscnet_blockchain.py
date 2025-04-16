import hashlib
import json
import time
import logging
import ecdsa

# Setup basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Token Definitions ---
MSC_TOKEN = "$MSC"    # Token for governance / advanced access
OMEGA_TOKEN = "Ω"     # Computational energy unit
PHI_TOKEN = "Φ"       # Impact metric (minted by epistemic consensus)

# --- Block Class with Proof of Synth ---
class Block:
    def __init__(self, index, timestamp, data, synth_proof, previous_hash, agent_signature=None):
        """
        agent_signature: digital signature of the block data produced by the agent.
        """
        self.index = index
        self.timestamp = timestamp
        self.data = data  # Contains synthesis proposals, evaluations, etc.
        self.synth_proof = synth_proof
        self.previous_hash = previous_hash
        self.agent_signature = agent_signature
        self.hash = self.calculate_hash()

    def calculate_hash(self):
        sha = hashlib.sha256()
        # Note: The block string excludes the agent_signature when computing the hash.
        block_string = f"{self.index}{self.timestamp}{json.dumps(self.data, sort_keys=True)}" \
                       f"{json.dumps(self.synth_proof, sort_keys=True)}{self.previous_hash}"
        sha.update(block_string.encode("utf-8"))
        return sha.hexdigest()

    def sign_block(self, signing_key):
        """
        Sign the block (using the block string used in calculate_hash).
        The signature is stored as hexadecimal in agent_signature.
        """
        block_string = f"{self.index}{self.timestamp}{json.dumps(self.data, sort_keys=True)}" \
                       f"{json.dumps(self.synth_proof, sort_keys=True)}{self.previous_hash}"
        signature = signing_key.sign(block_string.encode("utf-8"))
        self.agent_signature = signature.hex()

    def verify_signature(self, verifying_key):
        """
        Verifies the block's signature (if present) using the provided verifying key.
        """
        if not self.agent_signature:
            return False
        block_string = f"{self.index}{self.timestamp}{json.dumps(self.data, sort_keys=True)}" \
                       f"{json.dumps(self.synth_proof, sort_keys=True)}{self.previous_hash}"
        try:
            return verifying_key.verify(bytes.fromhex(self.agent_signature), block_string.encode("utf-8"))
        except Exception:
            return False

    def __repr__(self):
        return (f"Block(index={self.index}, time={self.timestamp:.2f}, "
                f"hash={self.hash[:10]}..., prev_hash={self.previous_hash[:10]}..., "
                f"synth_proof={self.synth_proof})")

# --- Modular Components ---

class SynthetixLayer:
    """
    Layer that manages propagation and storage of knowledge nodes.
    """
    def __init__(self):
        self.nodes = {}

    def add_node(self, node):
        # Assume node is a dict containing an 'id'
        self.nodes[node.get("id")] = node
        logging.debug(f"SynthetixLayer added node with id: {node.get('id')}")

class OmegaLedger:
    """
    Ledger for recording computational value transactions: Ω consumed/generated.
    """
    def __init__(self):
        self.transactions = []

    def record_transaction(self, tx):
        self.transactions.append(tx)
        logging.debug(f"OmegaLedger transaction recorded: {tx}")

class ReputationNet:
    """
    Epistemic reputation system (Ψ) based on cross-evaluation.
    """
    def __init__(self):
        self.reputations = {}

    def update_reputation(self, agent_id, delta):
        self.reputations[agent_id] = self.reputations.get(agent_id, 1.0) + delta
        logging.debug(f"ReputationNet: Agent {agent_id} updated by delta {delta}, total: {self.reputations[agent_id]}")

class KnowledgeVM:
    """
    Virtual machine to execute synthesis smart contracts.
    """
    def execute_contract(self, contract_data):
        # Aquí se debería validar cambios, actualizar reputaciones o modificar nodos.
        # Esta implementación es solo un ejemplo.
        logging.info("Executing contract in KnowledgeVM...")
        result = {"status": "success", "message": "Contract executed.", "details": contract_data}
        logging.debug(f"KnowledgeVM result: {result}")
        return result

class GraphState:
    """
    Maintains a compressed state of the global graph for verifying changes and consistency.
    """
    def __init__(self):
        self.state_summary = ""

    def update_state(self, state_data):
        self.state_summary = hashlib.sha256(json.dumps(state_data, sort_keys=True).encode("utf-8")).hexdigest()
        logging.debug(f"GraphState updated: {self.state_summary}")

    def evaluate_consistency_change(self, new_data):
        # Simulate a new state summary for new_data
        new_state = hashlib.sha256(json.dumps(new_data, sort_keys=True).encode("utf-8")).hexdigest()
        # Compute a dummy similarity score between 0 and 1 (for example, count matching hex digits)
        matches = sum(1 for a, b in zip(self.state_summary, new_state) if a == b)
        score = matches / len(new_state)
        logging.debug(f"Evaluated consistency change score: {score:.3f}")
        return score

# --- MSCNet Blockchain ---
class MSCNetBlockchain:
    def __init__(self, proof_thresholds=None):
        """
        proof_thresholds: dict for customizable proof parameters.
           Example: {"min_reputation": 0.5, "min_impact": 0.0}
        """
        # Default thresholds if not provided
        self.proof_thresholds = proof_thresholds or {"min_reputation": 0.5, "min_impact": 0.0}
        self.chain = [self.create_genesis_block()]
        # Initialize modular components
        self.synthetix = SynthetixLayer()
        self.omega_ledger = OmegaLedger()
        self.reputation_net = ReputationNet()
        self.knowledge_vm = KnowledgeVM()
        self.graph_state = GraphState()
        logging.info("MSCNetBlockchain initiated with genesis block.")

    def create_genesis_block(self):
        genesis_data = {"message": "Genesis Block for MSCNet", "details": {}}
        # Set neutral proof values in genesis
        synth_proof = {"Ψ": 1.0, "Φ": 0.0, "Ω": 0.0}
        genesis = Block(0, time.time(), genesis_data, synth_proof, "0")
        logging.info(f"Genesis block created: {genesis}")
        return genesis

    def get_latest_block(self):
        return self.chain[-1]

    def add_block(self, data, synth_proof, agent_signature=None):
        # Ejecuta el contrato (aplicando lógica de síntesis)
        vm_result = self.knowledge_vm.execute_contract(data)
        # Se podrían usar vm_result para modificar 'data' o agregar información
        last_block = self.get_latest_block()
        new_block = Block(last_block.index + 1, time.time(), data, synth_proof, last_block.hash, agent_signature)
        logging.info(f"Attempting to add block with index {new_block.index}")
        if self.validate_new_block(new_block, last_block):
            self.chain.append(new_block)
            logging.info(f"Block {new_block.index} successfully added.")
            return new_block
        else:
            error_msg = "Block validation failed: Proof of Synth did not meet requirements."
            logging.error(error_msg)
            raise Exception(error_msg)

    def validate_new_block(self, new_block, previous_block):
        # Check index and hash linking
        if previous_block.index + 1 != new_block.index:
            logging.error("Invalid index.")
            return False
        if previous_block.hash != new_block.previous_hash:
            logging.error("Hash linkage broken.")
            return False
        if new_block.calculate_hash() != new_block.hash:
            logging.error("Block hash recalculation mismatch.")
            return False
        # Validate Proof of Synth using thresholds
        proof = new_block.synth_proof
        if proof["Ψ"] < self.proof_thresholds["min_reputation"]:
            logging.error("Insufficient reputation (Ψ) in synth proof.")
            return False
        if proof["Φ"] < self.proof_thresholds["min_impact"]:
            logging.error("Negative impact (Φ) in synth proof.")
            return False

        # Evaluate consistency (Ω) using GraphState (dynamically computed delta)
        consistency_score = self.graph_state.evaluate_consistency_change(new_block.data)
        if proof["Ω"] < consistency_score:
            logging.warning("Inconsistent Ω compared to calculated consistency delta.")
            return False

        # If the block includes a signature, verify it.
        if new_block.agent_signature:
            agent_public_key_hex = new_block.data.get("agent_public_key")
            if not agent_public_key_hex:
                logging.error("Block has signature but no agent_public_key provided in data.")
                return False
            try:
                vk = ecdsa.VerifyingKey.from_string(bytes.fromhex(agent_public_key_hex), curve=ecdsa.SECP256k1)
                if not new_block.verify_signature(vk):
                    logging.error("Block signature verification failed.")
                    return False
            except Exception as e:
                logging.error(f"Error verifying signature: {e}")
                return False
        return True

    def is_chain_valid(self):
        for i in range(1, len(self.chain)):
            curr = self.chain[i]
            prev = self.chain[i - 1]
            if curr.hash != curr.calculate_hash():
                logging.error(f"Block {curr.index} hash invalid.")
                return False
            if curr.previous_hash != prev.hash:
                logging.error(f"Block {curr.index} previous hash does not match.")
                return False
        return True

    def export_chain(self, path="msc_chain.json"):
        with open(path, "w") as f:
            json.dump([block.__dict__ for block in self.chain], f, indent=2)
        logging.info(f"Exported blockchain to {path}")

    def __repr__(self):
        chain_data = [{"index": block.index, "hash": block.hash} for block in self.chain]
        return json.dumps(chain_data, indent=2)

# --- Example usage ---
if __name__ == "__main__":
    import argparse
    from wallet import Wallet  # Assuming wallet.py is in the workspace

    parser = argparse.ArgumentParser(description="MSCNet Blockchain CLI")
    subparsers = parser.add_subparsers(dest="command", help="Sub-command help")

    add_parser = subparsers.add_parser("add", help="Add a new block")
    add_parser.add_argument("--agent", type=str, required=True, help="Agent ID")
    add_parser.add_argument("--impact", type=float, required=True, help="Impact score (Φ)")
    add_parser.add_argument("--reputation", type=float, required=True, help="Reputation score (Ψ)")
    add_parser.add_argument("--consistency", type=float, required=True, help="Consistency score (Ω)")
    add_parser.add_argument("--action", type=str, required=True, help="Action description")

    export_parser = subparsers.add_parser("export", help="Export blockchain to JSON")
    export_parser.add_argument("--path", type=str, default="msc_chain.json", help="Export file path")

    args = parser.parse_args()

    custom_thresholds = {"min_reputation": 0.6, "min_impact": 0.1}
    mscnet = MSCNetBlockchain(proof_thresholds=custom_thresholds)

    if args.command == "add":
        # Create a wallet
        agent_wallet = Wallet()

        # Prepare the block data (include the agent's public key)
        transaction_data = {
            "agent_id": args.agent,
            "action": args.action,
            "content": f"Action executed by {args.agent}",
            "details": {},
            "agent_public_key": agent_wallet.get_public_key()
        }
        synth_proof = {"Ψ": args.reputation, "Φ": args.impact, "Ω": args.consistency}

        # Create the block
        last_block = mscnet.get_latest_block()
        new_block = Block(last_block.index + 1, time.time(), transaction_data, synth_proof, last_block.hash)
        # Sign the block with the agent's private key
        new_block.sign_block(agent_wallet.private_key)

        # Add the block to the blockchain (signature will be verified in validate_new_block)
        try:
            mscnet.add_block(transaction_data, synth_proof, new_block.agent_signature)
            logging.info("New block added to MSCNet:")
            logging.info(new_block)
        except Exception as e:
            logging.error("Error adding block:", exc_info=e)
    elif args.command == "export":
        mscnet.export_chain(args.path)
    else:
        parser.print_help()
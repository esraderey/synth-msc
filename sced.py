import numpy as np
import hashlib
import json
import time
import logging
from mscnet_blockchain import Block, MSCNetBlockchain

class EpistemicVector:
    """Representación multidimensional de una contribución epistémica"""
    
    DIMENSIONS = {
        "reputation": {"symbol": "Ψ", "weight": 0.30},
        "impact": {"symbol": "Φ", "weight": 0.25},
        "consistency": {"symbol": "Ω", "weight": 0.20},
        "novelty": {"symbol": "Ν", "weight": 0.15},
        "verification": {"symbol": "V", "weight": 0.10}
    }
    
    def __init__(self, values=None):
        """
        Initialize an epistemic vector with values for each dimension.
        If no values provided, initialize with default minimal values.
        """
        self.values = values or {
            "reputation": 0.5,      # Ψ: Reputation score
            "impact": 0.0,          # Φ: Impact score
            "consistency": 0.0,      # Ω: Consistency with existing knowledge
            "novelty": 0.0,         # Ν: Novelty of the contribution
            "verification": 0.0     # V: Verification level
        }
    
    def calculate_aggregate_score(self):
        """Calculate weighted aggregate score across all dimensions"""
        score = 0
        for dim, value in self.values.items():
            score += value * self.DIMENSIONS[dim]["weight"]
        return score
    
    def to_dict(self):
        """Convert to dictionary with symbols"""
        return {self.DIMENSIONS[dim]["symbol"]: value for dim, value in self.values.items()}
    
    @classmethod
    def from_dict(cls, symbol_dict):
        """Create EpistemicVector from dictionary with symbols"""
        # Reverse mapping from symbol to dimension name
        symbol_to_dim = {info["symbol"]: dim for dim, info in cls.DIMENSIONS.items()}
        values = {symbol_to_dim[symbol]: value for symbol, value in symbol_dict.items()}
        return cls(values)
    
    def __repr__(self):
        symbols = [f"{self.DIMENSIONS[dim]['symbol']}={value:.2f}" for dim, value in self.values.items()]
        return f"EpistemicVector({', '.join(symbols)})"


class DynamicThresholdPolicy:
    """Policy for dynamically adjusting validation thresholds based on context"""
    
    def __init__(self, base_thresholds=None):
        """Initialize with base thresholds that will be dynamically adjusted"""
        self.base_thresholds = base_thresholds or {
            "reputation": 0.5,
            "impact": 0.1,
            "consistency": 0.3,
            "novelty": 0.0,
            "verification": 0.2
        }
        self.domain_adjustments = {}  # Adjustments based on knowledge domain
        self.context_memory = []  # Recent context for adaptive adjustments
    
    def register_domain(self, domain_name, adjustments):
        """Register threshold adjustments for a specific knowledge domain"""
        self.domain_adjustments[domain_name] = adjustments
    
    def calculate_thresholds(self, domain=None, context=None):
        """Calculate effective thresholds based on domain and context"""
        # Start with base thresholds
        effective = dict(self.base_thresholds)
        
        # Apply domain-specific adjustments if applicable
        if domain and domain in self.domain_adjustments:
            for dim, adjustment in self.domain_adjustments[domain].items():
                effective[dim] = max(0.0, min(1.0, effective[dim] + adjustment))
        
        # Apply context-based adaptations (simplified implementation)
        if context and len(self.context_memory) > 0:
            # Example: If recent blocks have high novelty, increase novelty threshold
            avg_novelty = sum(ctx.get('novelty', 0) for ctx in self.context_memory) / len(self.context_memory)
            if avg_novelty > 0.7:  # High novelty environment
                effective['novelty'] = min(1.0, effective['novelty'] * 1.2)  # Increase novelty threshold
        
        return effective
    
    def update_context_memory(self, new_context):
        """Update the memory of recent contexts for adaptive adjustments"""
        self.context_memory.append(new_context)
        if len(self.context_memory) > 10:  # Keep last 10 contexts
            self.context_memory.pop(0)


class SCEDBlock(Block):
    """Enhanced block with SCED capabilities"""
    
    def __init__(self, index, timestamp, data, epistemic_vector, previous_hash, agent_signature=None,
                 supporting_signatures=None):
        # Convert epistemic vector to legacy synth_proof format for compatibility
        if isinstance(epistemic_vector, EpistemicVector):
            synth_proof = epistemic_vector.to_dict()
        else:
            synth_proof = epistemic_vector
            
        super().__init__(index, timestamp, data, synth_proof, previous_hash, agent_signature)
        
        # SCED-specific extensions
        self.supporting_signatures = supporting_signatures or []  # For multi-agent validation
        self.epistemic_vector = epistemic_vector if isinstance(epistemic_vector, EpistemicVector) else EpistemicVector.from_dict(synth_proof)
        self.validation_context = {}  # Store context used during validation
    
    def add_supporting_signature(self, agent_id, signature):
        """Add a supporting signature from a validating agent"""
        self.supporting_signatures.append({
            "agent_id": agent_id,
            "signature": signature,
            "timestamp": time.time()
        })
    
    def calculate_hash(self):
        """Override to include supporting signatures in hash calculation"""
        sha = hashlib.sha256()
        block_string = super().calculate_hash()  # Get the base hash
        
        # Add supporting signatures
        if self.supporting_signatures:
            block_string += json.dumps(self.supporting_signatures, sort_keys=True)
            
        sha.update(block_string.encode("utf-8"))
        return sha.hexdigest()
        
    def to_dict(self):
        """Convert to dict including SCED extensions"""
        block_dict = {
            "index": self.index,
            "timestamp": self.timestamp,
            "data": self.data,
            "synth_proof": self.synth_proof,
            "previous_hash": self.previous_hash,
            "hash": self.hash,
            "agent_signature": self.agent_signature,
            "supporting_signatures": self.supporting_signatures,
            "validation_context": self.validation_context
        }
        return block_dict


class SCEDConsensus:
    """Consensus mechanism implementing the SCED approach"""
    
    def __init__(self):
        self.threshold_policy = DynamicThresholdPolicy()
        self.validator_registry = {}  # Registry of validator weights and history
        
        # Register some example domain-specific threshold adjustments
        self.threshold_policy.register_domain("scientific_research", {
            "verification": 0.2,  # Higher verification needed
            "novelty": 0.1       # Higher novelty expected
        })
        
        self.threshold_policy.register_domain("operational_data", {
            "consistency": 0.2,   # Higher consistency required
            "verification": -0.1  # Lower verification threshold
        })
    
    def validate_epistemic_contribution(self, block, context=None):
        """
        Validate a block's epistemic contribution based on its vector
        and dynamically calculated thresholds.
        """
        # Extract domain from data if available
        domain = block.data.get("domain", None)
        
        # Calculate effective thresholds
        thresholds = self.threshold_policy.calculate_thresholds(domain, context)
        
        # Store validation context in the block
        block.validation_context = {
            "domain": domain,
            "applied_thresholds": thresholds,
            "validation_time": time.time()
        }
        
        # Check each dimension against threshold
        vector = block.epistemic_vector
        for dim, threshold in thresholds.items():
            if dim in vector.values and vector.values[dim] < threshold:
                logging.warning(f"Failed validation on dimension {dim}: {vector.values[dim]} < {threshold}")
                return False
        
        # Calculate aggregate score for additional validation
        aggregate_score = vector.calculate_aggregate_score()
        if aggregate_score < 0.4:  # Minimum aggregate threshold
            logging.warning(f"Failed validation on aggregate score: {aggregate_score} < 0.4")
            return False
            
        return True
    
    def register_validator(self, validator_id, weight=1.0, specializations=None):
        """Register a validator with optional domain specializations"""
        self.validator_registry[validator_id] = {
            "weight": weight,
            "specializations": specializations or [],
            "history": []
        }
    
    def record_validation(self, validator_id, block_hash, result):
        """Record a validation performed by a validator"""
        if validator_id in self.validator_registry:
            self.validator_registry[validator_id]["history"].append({
                "block_hash": block_hash,
                "result": result,
                "timestamp": time.time()
            })


class SCEDBlockchain(MSCNetBlockchain):
    """Extended blockchain implementing the SCED protocol"""
    
    def __init__(self, proof_thresholds=None):
        super().__init__(proof_thresholds)
        self.consensus = SCEDConsensus()
        self.domains = {}  # Domain-specific knowledge contexts
        logging.info("SCED Blockchain initiated with enhanced consensus mechanism")
    
    def create_genesis_block(self):
        """Create a genesis block with SCED capabilities"""
        genesis_data = {"message": "Genesis Block for SCED", "details": {}}
        # Initialize with neutral epistemic vector
        epistemic_vector = EpistemicVector()
        genesis = SCEDBlock(0, time.time(), genesis_data, epistemic_vector, "0")
        logging.info(f"SCED Genesis block created: {genesis}")
        return genesis
    
    def add_block(self, data, epistemic_vector, agent_signature=None):
        """Add a block using SCED consensus"""
        # Enrich with domain context if available
        if "domain" in data:
            domain_name = data["domain"]
            # Update domain-specific context
            if domain_name not in self.domains:
                self.domains[domain_name] = {"block_count": 0, "last_update": 0}
            self.domains[domain_name]["block_count"] += 1
            self.domains[domain_name]["last_update"] = time.time()
        
        # Create validation context from recent blocks
        context = {
            "recent_blocks": [self.chain[-i].epistemic_vector.values 
                              for i in range(1, min(6, len(self.chain)+1))],
            "domains": self.domains
        }
        
        # Ensure epistemic_vector is an EpistemicVector object
        if not isinstance(epistemic_vector, EpistemicVector):
            if isinstance(epistemic_vector, dict):
                epistemic_vector = EpistemicVector.from_dict(epistemic_vector)
            else:
                raise TypeError("epistemic_vector must be an EpistemicVector or compatible dict")
        
        # Create the new block
        last_block = self.get_latest_block()
        new_block = SCEDBlock(
            last_block.index + 1,
            time.time(),
            data,
            epistemic_vector,
            last_block.hash,
            agent_signature
        )
        
        # Validate using SCED consensus
        if self.consensus.validate_epistemic_contribution(new_block, context):
            self.chain.append(new_block)
            # Update context memory for threshold adaptation
            self.consensus.threshold_policy.update_context_memory({
                "domain": data.get("domain"),
                "novelty": epistemic_vector.values.get("novelty", 0),
                "timestamp": time.time()
            })
            logging.info(f"Block {new_block.index} successfully added with SCED validation.")
            return new_block
        else:
            error_msg = "Block validation failed: Did not meet SCED consensus requirements."
            logging.error(error_msg)
            raise Exception(error_msg)

    def export_chain(self, path="sced_chain.json"):
        """Export the SCED blockchain to a JSON file"""
        with open(path, "w") as f:
            chain_data = [block.to_dict() if hasattr(block, 'to_dict') else block.__dict__ for block in self.chain]
            json.dump(chain_data, f, indent=2)
        logging.info(f"Exported SCED blockchain to {path}")


# --- Example usage ---
if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    # Initialize SCED blockchain
    sced = SCEDBlockchain()
    
    # Register some validators
    sced.consensus.register_validator("expert_agent_1", weight=1.5, specializations=["scientific_research"])
    sced.consensus.register_validator("validator_2", weight=1.0, specializations=["operational_data"])
    
    # Create a sample epistemic vector
    vector = EpistemicVector({
        "reputation": 0.8,
        "impact": 0.6,
        "consistency": 0.7,
        "novelty": 0.5,
        "verification": 0.4
    })
    
    # Add a block using the SCED consensus
    try:
        block_data = {
            "agent_id": "expert_agent_1",
            "domain": "scientific_research",
            "action": "knowledge_contribution",
            "content": "New research findings on SCED technology",
            "details": {
                "references": ["paper1", "paper2"],
                "methodology": "experimental"
            }
        }
        
        new_block = sced.add_block(block_data, vector)
        print(f"Successfully added block: {new_block}")
        print(f"Block validation context: {new_block.validation_context}")
        
    except Exception as e:
        logging.error(f"Failed to add block: {e}")
    
    # Export the chain
    sced.export_chain("sced_example.json")
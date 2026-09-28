import hashlib
import time
import torch

class HashChainAudit:
    def __init__(self):
        self.chain = []
        # Genesis block
        self.chain.append({
            'hospital_id': 'genesis',
            'communication_round': 0,
            'timestamp': time.time(),
            'update_hash': '0',
            'previous_hash': '0'
        })
        
    def _hash_record(self, record):
        record_string = f"{record['hospital_id']}{record['communication_round']}{record['timestamp']}{record['update_hash']}{record['previous_hash']}"
        return hashlib.sha256(record_string.encode()).hexdigest()

    def create_audit_record(self, hospital_id, comm_round, update_state):
        # Create hash of the model update
        # We can hash the string representation of some key parameters or just serialize it
        # For efficiency, we hash the string of a few keys
        update_str = str([update_state[k].sum().item() for k in list(update_state.keys())[:5]])
        update_hash = hashlib.sha256(update_str.encode()).hexdigest()
        
        prev_block = self.chain[-1]
        prev_block_hash = self._hash_record(prev_block)
        
        record = {
            'hospital_id': hospital_id,
            'communication_round': comm_round,
            'timestamp': time.time(),
            'update_hash': update_hash,
            'previous_hash': prev_block_hash
        }
        self.chain.append(record)
        return record
        
    def verify_audit_chain(self):
        for i in range(1, len(self.chain)):
            current = self.chain[i]
            previous = self.chain[i-1]
            if current['previous_hash'] != self._hash_record(previous):
                return False
        return True

def run_tamper_test():
    import pandas as pd
    from config import RESULTS_DIR
    import os
    
    audit = HashChainAudit()
    state1 = {'layer': torch.tensor([1.0])}
    state2 = {'layer': torch.tensor([2.0])}
    
    audit.create_audit_record(1, 1, state1)
    audit.create_audit_record(2, 1, state2)
    
    results = []
    
    # Test 1: Original chain
    is_valid = audit.verify_audit_chain()
    results.append({'condition': 'Original chain', 'records_tested': len(audit.chain), 'tampering_detected': not is_valid})
    
    # Test 2: Modified metadata
    audit.chain[1]['hospital_id'] = 99
    is_valid = audit.verify_audit_chain()
    results.append({'condition': 'Modified metadata', 'records_tested': len(audit.chain), 'tampering_detected': not is_valid})
    
    # Test 3: Modified update hash
    audit.chain[1]['hospital_id'] = 1 # Revert
    audit.chain[1]['update_hash'] = 'fake_hash'
    is_valid = audit.verify_audit_chain()
    results.append({'condition': 'Modified update hash', 'records_tested': len(audit.chain), 'tampering_detected': not is_valid})
    
    df = pd.DataFrame(results)
    path = os.path.join(RESULTS_DIR, 'audit_results.csv')
    df.to_csv(path, index=False)
    print(f"Tamper test complete. Saved to {path}")
    
if __name__ == "__main__":
    import torch
    run_tamper_test()

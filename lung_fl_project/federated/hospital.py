from federated.aggregation import aggregate_models

class HospitalEdge:
    def __init__(self, hospital_id, departments):
        self.hospital_id = hospital_id
        self.departments = departments

    def aggregate_departments(self, global_model):
        """
        Runs local training for each department, then aggregates their updates.
        Returns the hospital-level update and total samples.
        """
        dept_updates = []
        dept_samples = []
        
        for dept in self.departments:
            print(f"  Training Department {dept.client_id} (Hospital {self.hospital_id})...")
            state, samples = dept.train(global_model)
            dept_updates.append(state)
            dept_samples.append(samples)
            
        print(f"  Aggregating at Hospital Edge {self.hospital_id}...")
        hospital_state = aggregate_models(global_model.state_dict(), dept_updates, dept_samples)
        total_hospital_samples = sum(dept_samples)
        
        return hospital_state, total_hospital_samples

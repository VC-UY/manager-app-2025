// types.ts
export interface User {
    id: string;
    username: string;
    email: string;
  }
  
  export interface Workflow {
    id: string;
    name: string;
    description: string;
    workflow_type: string;
    status: string;
    owner: string;
    created_at: string;
    updated_at: string;
    submitted_at: string | null;
    completed_at: string | null;
    priority: number;
    tags: string[];
    metadata: any;
  }
  
  export interface Task {
    id: string;
    name: string;
    description: string;
    status: string;
    workflow: string;
    workflow_name?: string;
    command: string;
    parameters: any;
    progress: number;
    created_at: string;
    start_time: string | null;
    end_time: string | null;
    required_resources: any;
    estimated_max_time: number;
    subtasks?: Task[];
    volunteer_count?: number;
    logs?: string;
  }
  
  export interface Volunteer {
    id: string;
    name: string;
    hostname: string;
    ip_address: string;
    last_ip_address: string;
    cpu_cores: number;
    ram_mb: number;
    disk_gb: number;
    gpu: string;
    available: boolean;
    status: string;
    last_seen: string;
    tags: string[];
    assigned_tasks_count?: number;
  }
  
  export interface VolunteerTask {
    id: string;
    volunteer: string;
    volunteer_name?: string;
    task: string;
    task_name?: string;
    assigned_at: string;
    started_at: string | null;
    completed_at: string | null;
    status: string;
    progress: number;
    result: any;
    error: string | null;
  }
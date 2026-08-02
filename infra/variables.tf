variable "region" {
  description = "AWS region"
  type        = string
  default     = "eu-west-2"
}

variable "instance_type" {
  description = "EC2 size (free-tier eligible; prod uses a hosted LLM, not local Ollama)"
  type        = string
  default     = "t3.micro"
}

variable "openai_api_key" {
  description = "OpenAI key for the deployed instance (prod LLM provider)"
  type        = string
  sensitive   = true
}

variable "ssh_public_key_path" {
  description = "Path to SSH public key for debugging access"
  type        = string
  default     = "~/.ssh/finagent-demo.pub"
}

variable "my_ip_cidr" {
  description = "Your IP in CIDR form for SSH access, e.g. 1.2.3.4/32"
  type        = string
}

variable "sec_user_agent" {
  description = "SEC fair-access User-Agent: 'AppName your@email.com'"
  type        = string
}

variable "api_key" {
  description = "API key for the deployed service"
  type        = string
  sensitive   = true
}

variable "repo_url" {
  description = "Public repo to deploy"
  type        = string
  default     = "https://github.com/notadityajoshi/financial-research-agent.git"
}
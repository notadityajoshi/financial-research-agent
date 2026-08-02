terraform {
  required_version = ">= 1.7"
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }
}

provider "aws" {
  region = var.region
}

data "aws_ami" "al2023" {
  most_recent = true
  owners      = ["amazon"]
  filter {
    name   = "name"
    values = ["al2023-ami-2023.*-x86_64"]
  }
}

resource "aws_key_pair" "demo" {
  key_name   = "finagent-demo"
  public_key = file(pathexpand(var.ssh_public_key_path))
}

resource "aws_security_group" "demo" {
  name        = "finagent-demo"
  description = "API public, SSH restricted to my IP"

  ingress {
    description = "API"
    from_port   = 8000
    to_port     = 8000
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"] # API-key auth guards the endpoints
  }

  ingress {
    description = "SSH"
    from_port   = 22
    to_port     = 22
    protocol    = "tcp"
    cidr_blocks = [var.my_ip_cidr]
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }
}

resource "aws_instance" "demo" {
  ami                    = data.aws_ami.al2023.id
  instance_type          = var.instance_type
  key_name               = aws_key_pair.demo.key_name
  vpc_security_group_ids = [aws_security_group.demo.id]

  root_block_device {
    volume_size = 30 # docker images + ollama models
    volume_type = "gp3"
  }

  user_data = templatefile("${path.module}/user_data.sh.tftpl", {
    repo_url       = var.repo_url
    sec_user_agent = var.sec_user_agent
    api_key        = var.api_key
    openai_api_key = var.openai_api_key
  })

  tags = {
    Name    = "finagent-demo"
    Project = "financial-research-agent"
    Purpose = "temporary-demo-destroy-after-use"
  }
}
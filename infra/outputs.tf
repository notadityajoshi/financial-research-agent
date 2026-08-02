output "public_ip" {
  value = aws_instance.demo.public_ip
}

output "api_url" {
  value = "http://${aws_instance.demo.public_ip}:8000"
}

output "ssh" {
  value = "ssh -i ~/.ssh/finagent-demo ec2-user@${aws_instance.demo.public_ip}"
}
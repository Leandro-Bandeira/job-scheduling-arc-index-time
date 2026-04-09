import requests
import logging


# Configuração da URL base da API
BASE_URL = "https://api.harumi.io"


def request_otp(email):
    """
    Solicita um token OTP via e-mail.
    """
    otp_request_url = f"{BASE_URL}/api/users/otp"
    otp_request_payload = {"email": email}
    
    response = requests.post(otp_request_url, json=otp_request_payload)
    logging.debug(f"OTP request response: {response.status_code}, {response.text}")
    
    if response.status_code == 200:
        return True
    else:
        logging.error(f"Erro ao solicitar OTP: {response.status_code}, {response.text}")
        return False

def verify_otp(email, token):
    """
    Verifica o token OTP fornecido pelo usuário.
    """
    otp_verify_url = f"{BASE_URL}/api/users/otp/verify"
    otp_verify_payload = {"email": email, "token": token}
    
    response = requests.post(otp_verify_url, json=otp_verify_payload)
    logging.debug(f"OTP verify response: {response.status_code}, {response.text}")
    
    if response.status_code == 200:
        response_data = response.json()
        logging.info(f"Usuário autenticado: {response_data}")
        return True, response_data
    else:
        logging.error(f"Erro ao verificar OTP: {response.status_code}, {response.text}")
        return False

def list_organizations(logged_user):
    """
    Lista as organizações do usuário autenticado.
    
    Parameters:
        logged_user (dict): Dicionário contendo informações do usuário logado, incluindo o access_token.
    
    Returns:
        tuple: (success (bool), response (dict or list))
    """
    organizations_url = f"{BASE_URL}/api/users/organizations"
    access_token = logged_user.get("access_token")  # Obtém o token de acesso do usuário logado
    
    headers = {
        "Authorization": f"Bearer {access_token}"  # Adiciona o token no cabeçalho
    }
    
    try:
        response = requests.get(organizations_url, headers=headers)
        logging.debug(f"Organizations response: {response.status_code}, {response.text}")
        
        if response.status_code == 200:
            return True, response.json()
        else:
            return False, response.json()
    except requests.RequestException as e:
        logging.error(f"Erro ao listar organizações: {e}")
        return False, {"error": str(e)}
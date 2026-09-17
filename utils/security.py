"""
安全校验工具（修复官方下架原因：路径穿越 + SSRF）

1. 路径穿越防护：本地路径经 realpath 规范化后必须位于允许根目录内，
   禁止绝对路径越权访问与 `..` 穿越。
2. SSRF 防护：URL 仅允许 http/https，域名解析后的所有 IP 必须在公网
   （禁止环回/内网/链路本地/保留/组播地址段），并覆盖重定向链。
"""
import ipaddress
import os
import socket
from typing import List, Tuple
from urllib.parse import urlparse

# 非公网地址段（环回 / 内网 / 链路本地 / CGNAT / 文档保留 / 组播 / 保留）
PRIVATE_NETWORKS: List[ipaddress._BaseNetwork] = [
    ipaddress.ip_network("0.0.0.0/8"),       # 本网络（未指定）
    ipaddress.ip_network("10.0.0.0/8"),      # 私网
    ipaddress.ip_network("100.64.0.0/10"),   # CGNAT
    ipaddress.ip_network("127.0.0.0/8"),     # 环回
    ipaddress.ip_network("169.254.0.0/16"),  # 链路本地
    ipaddress.ip_network("172.16.0.0/12"),   # 私网
    ipaddress.ip_network("192.0.0.0/24"),    # IETF 协议保留
    ipaddress.ip_network("192.0.2.0/24"),    # TEST-NET-1
    ipaddress.ip_network("192.168.0.0/16"),  # 私网
    ipaddress.ip_network("198.18.0.0/15"),   # 基准测试
    ipaddress.ip_network("198.51.100.0/24"), # TEST-NET-2
    ipaddress.ip_network("203.0.113.0/24"),  # TEST-NET-3
    ipaddress.ip_network("224.0.0.0/4"),     # 组播
    ipaddress.ip_network("240.0.0.0/4"),     # 保留
    ipaddress.ip_network("255.255.255.255/32"),
    ipaddress.ip_network("::/128"),          # 未指定
    ipaddress.ip_network("::1/128"),         # 环回
    ipaddress.ip_network("fc00::/7"),        # IPv6 唯一本地地址
    ipaddress.ip_network("fe80::/10"),       # IPv6 链路本地
    ipaddress.ip_network("ff00::/8"),        # IPv6 组播
]

MAX_REDIRECTS = 5


def is_private_ip(ip_str: str) -> bool:
    """判断 IP 是否属于非公网地址段。解析失败视为不安全。"""
    try:
        ip = ipaddress.ip_address(ip_str.split("%")[0])
    except ValueError:
        return True
    return any(ip in net for net in PRIVATE_NETWORKS)


def _resolve_hostname(hostname: str) -> List[str]:
    """解析域名，返回全部 IP。解析失败返回空列表。"""
    try:
        infos = socket.getaddrinfo(
            hostname, None, proto=socket.IPPROTO_TCP
        )
        return sorted({info[4][0] for info in infos})
    except socket.gaierror:
        return []


def check_url_safety(url: str) -> Tuple[bool, str]:
    """
    SSRF 防护：校验 URL 是否可安全访问。
    返回 (是否安全, 错误信息)。安全时错误信息为空字符串。
    """
    try:
        parsed = urlparse(url)
    except ValueError:
        return False, "URL 格式无效"

    if parsed.scheme not in ("http", "https"):
        return False, f"仅支持 http/https 协议，收到: {parsed.scheme or '无协议'}"

    hostname = parsed.hostname
    if not hostname:
        return False, "URL 缺少主机名"

    # 直接 IP 形式：立即检查
    try:
        ip = ipaddress.ip_address(hostname)
        if is_private_ip(str(ip)):
            return False, f"目标地址 {hostname} 属于内网/保留地址段，已阻止"
        return True, ""
    except ValueError:
        pass  # 域名形式，继续解析

    ips = _resolve_hostname(hostname)
    if not ips:
        return False, f"无法解析域名: {hostname}"

    for ip in ips:
        if is_private_ip(ip):
            return False, f"目标地址 {ip}（{hostname}）属于内网/保留地址段，已阻止"

    return True, ""


def validate_local_path(path: str, allowed_roots: List[str]) -> Tuple[bool, str]:
    """
    路径穿越防护：校验本地路径。
    path 经 expanduser + realpath 规范化后，必须位于某个 allowed_roots 内
    （相等或为其子路径）。返回 (是否允许, 规范化后的真实路径或错误信息)。
    """
    if not path:
        return False, "路径为空"

    try:
        real_path = os.path.realpath(os.path.expanduser(path))
    except (OSError, ValueError) as e:
        return False, f"路径解析失败: {e}"

    for root in allowed_roots:
        if not root:
            continue
        real_root = os.path.realpath(os.path.expanduser(root))
        if real_path == real_root or real_path.startswith(real_root + os.sep):
            return True, real_path

    return False, "路径不在允许的目录内（仅限知识库数据目录）"


def get_allowed_local_roots(
    data_root: str, extra_dirs: str = ""
) -> List[str]:
    """
    计算允许的本地路径根目录：数据目录 + 配置的额外白名单目录（逗号分隔）。
    """
    roots = [data_root]
    if extra_dirs:
        for d in extra_dirs.split(","):
            d = d.strip()
            if d:
                roots.append(d)
    return roots

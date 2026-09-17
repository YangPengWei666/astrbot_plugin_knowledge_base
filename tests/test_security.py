# -*- coding: utf-8 -*-
"""安全加固本地回归测试（安全模块 + 下载链路）

运行：python tests/test_security.py
覆盖：
  A. SSRF：公网 URL 放行；内网/环回/保留段/云元数据/非 http(s) 拒绝
  B. 路径穿越：数据目录内放行；../ 穿越、绝对路径越权拒绝；白名单目录放行
  C. 下载链路：内网 URL 在 download_file 层被拒（不会实际发起连接）
"""
import asyncio
import os
import sys
import tempfile

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(REPO_ROOT))  # 父目录，使包名可导入
sys.path.insert(0, REPO_ROOT)

from astrbot_plugin_knowledge_base.utils.security import (  # noqa: E402
    check_url_safety,
    validate_local_path,
    get_allowed_local_roots,
    is_private_ip,
)


PASS = 0
FAIL = 0


def check(name: str, cond: bool, detail: str = ""):
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  ✅ {name}")
    else:
        FAIL += 1
        print(f"  ❌ {name} {detail}")


def test_ssrf():
    print("\n== A. SSRF 防护 ==")
    # 公网放行
    safe, _ = check_url_safety("https://example.com/file.txt")
    check("公网 https 放行", safe)
    safe, _ = check_url_safety("http://example.com/a.txt")
    check("公网 http 放行", safe)
    safe, _ = check_url_safety("https://raw.githubusercontent.com/AstrBotDevs/AstrBot/main/README.md")
    # 注：本沙箱 DNS 将 raw.githubusercontent.com 劫持到内网 192.168.255.186，
    # 安全校验应拦截（防 DNS rebinding）。真实服务器解析为公网 IP 时正常放行。
    check("DNS 劫持到内网 IP 被拦（防 rebinding）", not safe)

    # 内网/环回/保留段拒绝
    for url, tag in [
        ("http://127.0.0.1:8080/x", "IPv4 环回"),
        ("http://localhost/x", "localhost"),
        ("http://10.0.0.5/x", "10.0.0.0/8"),
        ("http://172.16.0.5/x", "172.16.0.0/12"),
        ("http://192.168.1.1/x", "192.168.0.0/16"),
        ("http://169.254.169.254/latest/meta-data", "云元数据 169.254.169.254"),
        ("http://0.0.0.0/x", "0.0.0.0/8"),
        ("http://100.64.0.1/x", "CGNAT 100.64/10"),
        ("http://[::1]/x", "IPv6 环回 ::1"),
        ("http://[fc00::1]/x", "IPv6 ULA"),
        ("http://224.0.0.1/x", "组播"),
    ]:
        safe, reason = check_url_safety(url)
        check(f"{tag} 拒绝 ({url})", not safe, f"实际放行了! {reason}")

    # 协议限制
    safe, _ = check_url_safety("ftp://example.com/x")
    check("ftp 协议拒绝", not safe)
    safe, _ = check_url_safety("file:///etc/passwd")
    check("file:// 协议拒绝", not safe)
    safe, _ = check_url_safety("javascript:alert(1)")
    check("javascript: 协议拒绝", not safe)
    safe, _ = check_url_safety("http://")
    check("无主机名拒绝", not safe)

    # 工具函数
    check("is_private_ip(127.0.0.1)", is_private_ip("127.0.0.1"))
    check("is_private_ip(8.8.8.8) 为公网", not is_private_ip("8.8.8.8"))
    check("is_private_ip(::1)", is_private_ip("::1"))


def test_path_traversal():
    print("\n== B. 路径穿越防护 ==")
    base = "/tmp/kb_data"
    os.makedirs(os.path.join(base, "sub"), exist_ok=True)
    os.makedirs("/tmp/kb_whitelist", exist_ok=True)
    try:
        with open(os.path.join(base, "ok.txt"), "w") as f:
            f.write("ok")
        with open(os.path.join(base, "sub", "ok2.txt"), "w") as f:
            f.write("ok")
        with open("/tmp/secret_host_file.txt", "w") as f:
            f.write("secret")

        roots = get_allowed_local_roots(base, "/tmp/kb_whitelist")

        # 放行
        ok, p = validate_local_path("/tmp/kb_data/ok.txt", roots)
        check("数据目录内文件放行", ok, p)
        ok, p = validate_local_path("/tmp/kb_data/sub/ok2.txt", roots)
        check("数据目录子目录放行", ok, p)
        ok, p = validate_local_path("/tmp/kb_whitelist/x.txt", roots)
        check("白名单目录放行", ok, p)

        # 拒绝
        ok, p = validate_local_path("/tmp/secret_host_file.txt", roots)
        check("数据目录外绝对路径拒绝", not ok, f"放行了! {p}")
        ok, p = validate_local_path("/tmp/kb_data/../secret_host_file.txt", roots)
        check(".. 穿越拒绝", not ok, f"放行了! {p}")
        ok, p = validate_local_path("/etc/passwd", roots)
        check("/etc/passwd 拒绝", not ok, f"放行了! {p}")
        ok, p = validate_local_path("../../etc/passwd", roots)
        check("相对 .. 穿越拒绝", not ok, f"放行了! {p}")
    finally:
        os.remove(os.path.join(base, "ok.txt"))
        os.remove(os.path.join(base, "sub", "ok2.txt"))
        os.remove("/tmp/secret_host_file.txt")
        os.rmdir(os.path.join(base, "sub"))
        os.rmdir(base)
        os.rmdir("/tmp/kb_whitelist")


async def test_download_ssrf():
    print("\n== C. 下载链路 SSRF（download_file 层） ==")
    from astrbot_plugin_knowledge_base.utils.file_utils import download_file

    with tempfile.TemporaryDirectory() as tmp:
        # 内网 URL：download_file 应在发起连接前被拦截并返回 None
        r = await download_file("http://127.0.0.1:9/x.txt", tmp)
        check("download_file 拒绝内网 URL", r is None, f"下载成功了! {r}")
        r = await download_file("http://169.254.169.254/latest/meta-data", tmp)
        check("download_file 拒绝云元数据 URL", r is None, f"下载成功了! {r}")
        r = await download_file("ftp://example.com/x.txt", tmp)
        check("download_file 拒绝非 http(s)", r is None, f"下载成功了! {r}")


def main():
    test_ssrf()
    test_path_traversal()
    asyncio.run(test_download_ssrf())
    print(f"\n==== 结果: {PASS} 通过 / {FAIL} 失败 ====")
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()

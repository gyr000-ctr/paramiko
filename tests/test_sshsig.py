# Copyright (C) 2026  Paramiko contributors
#
# This file is part of paramiko.
#
# Paramiko is free software; you can redistribute it and/or modify it under the
# terms of the GNU Lesser General Public License as published by the Free
# Software Foundation; either version 2.1 of the License, or (at your option)
# any later version.
#
# Paramiko is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR
# A PARTICULAR PURPOSE.  See the GNU Lesser General Public License for more
# details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with Paramiko; if not, write to the Free Software Foundation, Inc.,
# 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301 USA.

"""
Unit tests for the sshsig (SSH signature) implementation.
"""

import base64
import io
import os
import shutil
import subprocess
import tempfile
import unittest

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import ed25519, rsa

from paramiko import Ed25519Key, RSAKey
from paramiko.sshsig import (
    ARMORED_BEGIN,
    ARMORED_END,
    SSHSignature,
    VerificationStatus,
    sign_data,
    verify_signature,
)
from paramiko.ssh_exception import (
    SSHSigFormatError,
    SSHSigKeyMismatchError,
    SSHSigNamespaceError,
    SSHSigVerificationError,
)

MESSAGE = b"release artifact v5.1.0\n"


def _rsa_key():
    priv = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    pem = priv.private_bytes(
        serialization.Encoding.PEM,
        serialization.PrivateFormat.OpenSSH,
        serialization.NoEncryption(),
    )
    return RSAKey.from_private_key(io.StringIO(pem.decode("ascii"))), pem


def _ed25519_key():
    priv = ed25519.Ed25519PrivateKey.generate()
    pem = priv.private_bytes(
        serialization.Encoding.PEM,
        serialization.PrivateFormat.OpenSSH,
        serialization.NoEncryption(),
    )
    return Ed25519Key.from_private_key(io.StringIO(pem.decode("ascii"))), pem


class RoundTripTest(unittest.TestCase):
    def test_ed25519_armored_roundtrip(self):
        key, _ = _ed25519_key()
        sig = sign_data(key, MESSAGE, namespace="file")
        parsed = SSHSignature.from_armored(sig.to_armored())
        self.assertEqual(parsed.namespace, "file")
        self.assertEqual(parsed.hash_algorithm, "sha512")
        self.assertEqual(
            parsed.public_key.asbytes(), key.asbytes()
        )
        self.assertTrue(parsed.verify(MESSAGE, key=key, namespace="file"))

    def test_ed25519_binary_roundtrip(self):
        key, _ = _ed25519_key()
        sig = sign_data(key, MESSAGE)
        parsed = SSHSignature.from_bytes(sig.to_bytes())
        self.assertTrue(parsed.verify(MESSAGE))

    def test_rsa_sha2_512_roundtrip(self):
        key, _ = _rsa_key()
        sig = sign_data(key, MESSAGE)
        parsed = SSHSignature.load(sig.to_armored())
        self.assertTrue(parsed.verify(MESSAGE, key=key, namespace="file"))

    def test_rsa_sha2_256_roundtrip(self):
        key, _ = _rsa_key()
        sig = sign_data(key, MESSAGE, sign_algorithm="rsa-sha2-256")
        parsed = SSHSignature.load(sig.to_bytes())
        self.assertTrue(parsed.verify(MESSAGE))

    def test_sha256_hash_algorithm_roundtrip(self):
        key, _ = _ed25519_key()
        sig = sign_data(key, MESSAGE, hash_algorithm="sha256")
        parsed = SSHSignature.load(sig.to_armored())
        self.assertEqual(parsed.hash_algorithm, "sha256")
        self.assertTrue(parsed.verify(MESSAGE))

    def test_custom_namespace_roundtrip(self):
        key, _ = _ed25519_key()
        sig = sign_data(key, MESSAGE, namespace="release-artifacts")
        parsed = SSHSignature.load(sig.to_armored())
        self.assertTrue(parsed.verify(MESSAGE, namespace="release-artifacts"))

    def test_load_autodetects_armored_bytes_and_text(self):
        key, _ = _ed25519_key()
        sig = sign_data(key, MESSAGE)
        armored = sig.to_armored()
        for form in (armored, armored.encode("ascii"), sig.to_bytes(), sig):
            parsed = (
                form
                if isinstance(form, SSHSignature)
                else SSHSignature.load(form)
            )
            self.assertTrue(parsed.verify(MESSAGE))


class FormatErrorTest(unittest.TestCase):
    def test_garbage_bytes(self):
        with self.assertRaises(SSHSigFormatError):
            SSHSignature.from_bytes(b"not a signature at all")

    def test_bad_magic(self):
        key, _ = _ed25519_key()
        blob = b"XXXXXX" + sign_data(key, MESSAGE).to_bytes()[6:]
        with self.assertRaises(SSHSigFormatError):
            SSHSignature.from_bytes(blob)

    def test_bad_version(self):
        key, _ = _ed25519_key()
        blob = bytearray(sign_data(key, MESSAGE).to_bytes())
        blob[9] = 99  # version uint32, big-endian last byte
        with self.assertRaises(SSHSigFormatError):
            SSHSignature.from_bytes(bytes(blob))

    def test_truncated_blob(self):
        key, _ = _ed25519_key()
        blob = sign_data(key, MESSAGE).to_bytes()
        with self.assertRaises(SSHSigFormatError):
            SSHSignature.from_bytes(blob[:20])

    def test_trailing_garbage(self):
        key, _ = _ed25519_key()
        blob = sign_data(key, MESSAGE).to_bytes() + b"extra"
        with self.assertRaises(SSHSigFormatError):
            SSHSignature.from_bytes(blob)

    def test_missing_armor(self):
        with self.assertRaises(SSHSigFormatError):
            SSHSignature.from_armored("-----BEGIN SOMETHING-----\nAAAA\n")

    def test_invalid_base64(self):
        with self.assertRaises(SSHSigFormatError):
            SSHSignature.from_armored(
                "{}\n!!! not base64 !!!\n{}\n".format(
                    ARMORED_BEGIN, ARMORED_END
                )
            )

    def test_armored_binary_mismatch(self):
        # Valid armor around a non-sshsig blob
        body = base64.b64encode(b"\x00" * 32).decode("ascii")
        with self.assertRaises(SSHSigFormatError):
            SSHSignature.from_armored(
                "{}\n{}\n{}\n".format(ARMORED_BEGIN, body, ARMORED_END)
            )

    def test_unsupported_hash_algorithm_on_sign(self):
        key, _ = _ed25519_key()
        with self.assertRaises(SSHSigFormatError):
            sign_data(key, MESSAGE, hash_algorithm="md5")

    def test_non_bytes_input(self):
        with self.assertRaises(SSHSigFormatError):
            SSHSignature.from_bytes("SSHSIG")


class VerificationFailureTest(unittest.TestCase):
    def test_namespace_mismatch(self):
        key, _ = _ed25519_key()
        sig = sign_data(key, MESSAGE, namespace="file")
        with self.assertRaises(SSHSigNamespaceError):
            sig.verify(MESSAGE, namespace="git")

    def test_key_mismatch(self):
        key, _ = _ed25519_key()
        other, _ = _ed25519_key()
        sig = sign_data(key, MESSAGE)
        with self.assertRaises(SSHSigKeyMismatchError):
            sig.verify(MESSAGE, key=other)

    def test_bad_signature_on_tampered_data(self):
        key, _ = _ed25519_key()
        sig = sign_data(key, MESSAGE)
        with self.assertRaises(SSHSigVerificationError):
            sig.verify(MESSAGE + b"tampered")

    def test_bad_signature_cross_key(self):
        # Signature bytes from a different key must not verify.
        key, _ = _ed25519_key()
        other, _ = _ed25519_key()
        sig = sign_data(key, MESSAGE)
        sig.public_key = other
        with self.assertRaises(SSHSigVerificationError):
            sig.verify(MESSAGE)


class VerifySignatureResultTest(unittest.TestCase):
    def setUp(self):
        self.key, _ = _ed25519_key()
        self.sig = sign_data(self.key, MESSAGE)

    def test_valid(self):
        result = verify_signature(
            MESSAGE, self.sig.to_armored(), trusted_keys=[self.key]
        )
        self.assertEqual(result.status, VerificationStatus.VALID)
        self.assertTrue(result.ok)
        self.assertTrue(result)
        self.assertIsNone(result.error)
        self.assertIsNotNone(result.signature)

    def test_valid_without_trusted_keys(self):
        result = verify_signature(MESSAGE, self.sig.to_bytes())
        self.assertEqual(result.status, VerificationStatus.VALID)

    def test_untrusted_key(self):
        other, _ = _ed25519_key()
        result = verify_signature(
            MESSAGE, self.sig.to_armored(), trusted_keys=[other]
        )
        self.assertEqual(result.status, VerificationStatus.UNTRUSTED_KEY)
        self.assertFalse(result.ok)
        self.assertIsNotNone(result.error)

    def test_unparseable(self):
        result = verify_signature(MESSAGE, b"\xde\xad\xbe\xef")
        self.assertEqual(result.status, VerificationStatus.UNPARSEABLE)
        self.assertFalse(result.ok)
        self.assertIsNone(result.signature)
        self.assertIsNotNone(result.error)

    def test_namespace_mismatch(self):
        result = verify_signature(
            MESSAGE,
            self.sig.to_armored(),
            trusted_keys=[self.key],
            namespace="git",
        )
        self.assertEqual(result.status, VerificationStatus.NAMESPACE_MISMATCH)

    def test_bad_signature(self):
        result = verify_signature(
            MESSAGE + b"tampered",
            self.sig.to_armored(),
            trusted_keys=[self.key],
        )
        self.assertEqual(result.status, VerificationStatus.BAD_SIGNATURE)


@unittest.skipUnless(shutil.which("ssh-keygen"), "ssh-keygen not available")
class OpenSSHInteropTest(unittest.TestCase):
    """
    Interop with OpenSSH: signatures we produce must verify with
    ``ssh-keygen -Y verify`` and signatures OpenSSH produces must verify
    with us.
    """

    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="paramiko-sshsig-test-")
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)
        self.data_file = os.path.join(self.tmp, "data.txt")
        with open(self.data_file, "wb") as f:
            f.write(MESSAGE)

    def _write_key(self, name, pem, key):
        key_path = os.path.join(self.tmp, name)
        with open(key_path, "wb") as f:
            f.write(pem)
        os.chmod(key_path, 0o600)
        pub_line = "{} {}\n".format(
            key.get_name(),
            base64.b64encode(key.asbytes()).decode("ascii"),
        )
        signers = os.path.join(self.tmp, name + ".allowed_signers")
        with open(signers, "w") as f:
            f.write("test-identity " + pub_line)
        return key_path, signers

    def _openssh_verify(self, signers, sig_path):
        proc = subprocess.run(
            [
                "ssh-keygen",
                "-Y",
                "verify",
                "-f",
                signers,
                "-I",
                "test-identity",
                "-n",
                "file",
                "-s",
                sig_path,
            ],
            input=MESSAGE,
            capture_output=True,
        )
        self.assertEqual(
            proc.returncode,
            0,
            "ssh-keygen -Y verify failed: {}{}".format(
                proc.stdout, proc.stderr
            ),
        )

    def _paramiko_signs_openssh_verifies(self, key_factory, name):
        key, pem = key_factory()
        _, signers = self._write_key(name, pem, key)
        sig_path = os.path.join(self.tmp, name + ".sig")
        with open(sig_path, "w") as f:
            f.write(sign_data(key, MESSAGE, namespace="file").to_armored())
        self._openssh_verify(signers, sig_path)

    def _openssh_signs_paramiko_verifies(self, key_factory, name):
        key, pem = key_factory()
        key_path, _ = self._write_key(name, pem, key)
        subprocess.run(
            ["ssh-keygen", "-Y", "sign", "-f", key_path, "-n", "file",
             self.data_file],
            check=True,
            capture_output=True,
        )
        with open(self.data_file + ".sig", "rb") as f:
            armored = f.read()
        result = verify_signature(
            MESSAGE, armored, trusted_keys=[key], namespace="file"
        )
        self.assertEqual(
            result.status,
            VerificationStatus.VALID,
            "paramiko rejected OpenSSH signature: {}".format(result.error),
        )

    def test_paramiko_ed25519_sig_verifies_with_openssh(self):
        self._paramiko_signs_openssh_verifies(_ed25519_key, "ed25519")

    def test_paramiko_rsa_sig_verifies_with_openssh(self):
        self._paramiko_signs_openssh_verifies(_rsa_key, "rsa")

    def test_openssh_ed25519_sig_verifies_with_paramiko(self):
        self._openssh_signs_paramiko_verifies(_ed25519_key, "ed25519")

    def test_openssh_rsa_sig_verifies_with_paramiko(self):
        self._openssh_signs_paramiko_verifies(_rsa_key, "rsa")


if __name__ == "__main__":
    unittest.main()

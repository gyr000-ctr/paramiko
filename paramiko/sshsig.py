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
SSH signatures (the OpenSSH ``sshsig`` format).

This module implements signing and verification of arbitrary data using SSH
keys, as described in OpenSSH's ``PROTOCOL.sshsig`` and produced/consumed by
``ssh-keygen -Y sign`` / ``ssh-keygen -Y verify``.

Signatures can be parsed from (and serialized to) both the binary blob form
and the ASCII-armored form::

    -----BEGIN SSH SIGNATURE-----
    ...
    -----END SSH SIGNATURE-----

Two API levels are provided:

- `SSHSignature` plus `sign_data`: raising APIs. Parsing failures raise
  `.SSHSigFormatError`; verification failures raise one of
  `.SSHSigNamespaceError`, `.SSHSigKeyMismatchError` or
  `.SSHSigVerificationError` depending on the cause. Nothing is swallowed.
- `verify_signature`: a policy-friendly wrapper returning a
  `VerificationResult` whose ``status`` (a `VerificationStatus`) distinguishes
  valid signatures, untrusted keys, unparseable input, namespace mismatches
  and bad signatures, without raising.

.. versionadded:: 5.2
"""

import base64
import binascii
import enum
import hashlib
import struct

from paramiko.message import Message
from paramiko.pkey import PKey
from paramiko.ssh_exception import (
    SSHException,
    SSHSigFormatError,
    SSHSigKeyMismatchError,
    SSHSigNamespaceError,
    SSHSigVerificationError,
)

#: Magic preamble at the start of every sshsig blob and signed preimage.
MAGIC_PREAMBLE = b"SSHSIG"

#: Only version of the sshsig format currently defined by OpenSSH.
SIG_VERSION = 1

#: Armor header/footer lines for the PEM-like text representation.
ARMORED_BEGIN = "-----BEGIN SSH SIGNATURE-----"
ARMORED_END = "-----END SSH SIGNATURE-----"

#: Hash algorithms OpenSSH defines for sshsig, mapped to hashlib constructors.
HASH_ALGORITHMS = {
    "sha256": hashlib.sha256,
    "sha512": hashlib.sha512,
}

#: Namespace OpenSSH defaults to when signing files.
DEFAULT_NAMESPACE = "file"

#: Line wrap width used by OpenSSH for the armored form.
_ARMOR_LINE_WIDTH = 70


class VerificationStatus(enum.Enum):
    """
    Outcome categories returned by `verify_signature`.

    Intended for policy layers: ``VALID`` means cryptographically sound and
    bound to a trusted key and the expected namespace; ``UNTRUSTED_KEY``
    means the embedded public key is not one the caller trusts;
    ``UNPARSEABLE`` means the input was not a well-formed sshsig blob;
    ``NAMESPACE_MISMATCH`` and ``BAD_SIGNATURE`` cover the remaining
    failure modes.

    .. versionadded:: 5.2
    """

    VALID = "valid"
    UNTRUSTED_KEY = "untrusted-key"
    UNPARSEABLE = "unparseable"
    NAMESPACE_MISMATCH = "namespace-mismatch"
    BAD_SIGNATURE = "bad-signature"


class VerificationResult:
    """
    Structured, non-raising result of `verify_signature`.

    Attributes:

    - ``status``: a `VerificationStatus` member.
    - ``signature``: the parsed `SSHSignature`, or ``None`` if the input
      could not be parsed.
    - ``error``: the exception that caused a failure status, or ``None``
      when ``status`` is ``VALID``. Never swallowed — always inspectable.

    .. versionadded:: 5.2
    """

    def __init__(self, status, signature=None, error=None):
        self.status = status
        self.signature = signature
        self.error = error

    @property
    def ok(self):
        """Whether the signature verified against a trusted key."""
        return self.status is VerificationStatus.VALID

    def __bool__(self):
        return self.ok

    def __repr__(self):
        return "<VerificationResult {}>".format(self.status.value)


class SSHSignature:
    """
    A parsed SSH signature (sshsig) blob.

    Attributes:

    - ``public_key``: the `.PKey` subclass instance embedded in the blob.
    - ``namespace``: the signature namespace (``str``), e.g. ``"file"``.
    - ``hash_algorithm``: ``"sha256"`` or ``"sha512"``.
    - ``signature``: raw bytes of the inner signature blob (itself an SSH
      ``string(algorithm) + string(signature)`` pair).
    - ``reserved``: the (currently always empty) reserved field, as bytes.

    .. versionadded:: 5.2
    """

    def __init__(
        self,
        public_key,
        namespace,
        hash_algorithm,
        signature,
        reserved=b"",
    ):
        if hash_algorithm not in HASH_ALGORITHMS:
            raise SSHSigFormatError(
                "unsupported hash algorithm: {!r}".format(hash_algorithm)
            )
        self.public_key = public_key
        self.namespace = namespace
        self.hash_algorithm = hash_algorithm
        self.signature = signature
        self.reserved = reserved

    @classmethod
    def from_bytes(cls, data):
        """
        Parse a binary sshsig blob.

        :param bytes data: the raw blob (starting with ``SSHSIG``).
        :raises SSHSigFormatError: on any structural problem.
        """
        if not isinstance(data, (bytes, bytearray)):
            raise SSHSigFormatError(
                "expected bytes, got {}".format(type(data).__name__)
            )
        data = bytes(data)
        if not data.startswith(MAGIC_PREAMBLE):
            raise SSHSigFormatError("missing SSHSIG magic preamble")
        try:
            (version,) = struct.unpack(
                ">I", data[len(MAGIC_PREAMBLE) : len(MAGIC_PREAMBLE) + 4]
            )
        except struct.error:
            raise SSHSigFormatError("truncated signature version")
        if version != SIG_VERSION:
            raise SSHSigFormatError(
                "unsupported signature version: {}".format(version)
            )
        msg = Message(data[len(MAGIC_PREAMBLE) + 4 :])
        try:
            # Wire order per PROTOCOL.sshsig: publickey, namespace, reserved,
            # hash_algorithm, signature.
            key_blob = msg.get_binary()
            namespace = msg.get_text()
            reserved = msg.get_binary()
            hash_algorithm = msg.get_text()
            signature = msg.get_binary()
        except (SSHException, UnicodeDecodeError) as e:
            raise SSHSigFormatError(
                "malformed signature blob: {}".format(e)
            ) from e
        remainder = msg.get_remainder()
        if remainder:
            raise SSHSigFormatError("trailing data after signature blob")
        if reserved:
            raise SSHSigFormatError("reserved field must be empty")
        if hash_algorithm not in HASH_ALGORITHMS:
            raise SSHSigFormatError(
                "unsupported hash algorithm: {!r}".format(hash_algorithm)
            )
        public_key = _parse_public_key(key_blob)
        return cls(
            public_key=public_key,
            namespace=namespace,
            hash_algorithm=hash_algorithm,
            signature=signature,
            reserved=reserved,
        )

    @classmethod
    def from_armored(cls, text):
        """
        Parse an ASCII-armored sshsig signature.

        :param str text: text including the ``BEGIN/END SSH SIGNATURE``
            lines (``bytes`` input is also accepted and decoded as ASCII).
        :raises SSHSigFormatError: on missing armor or invalid base64.
        """
        if isinstance(text, (bytes, bytearray)):
            try:
                text = bytes(text).decode("ascii")
            except UnicodeDecodeError as e:
                raise SSHSigFormatError(
                    "armored signature is not ASCII"
                ) from e
        lines = [line.strip() for line in text.strip().splitlines()]
        try:
            begin = lines.index(ARMORED_BEGIN)
            end = lines.index(ARMORED_END)
        except ValueError:
            raise SSHSigFormatError(
                "missing SSH SIGNATURE armor header/footer"
            )
        if end <= begin:
            raise SSHSigFormatError("armor footer precedes header")
        body = "".join(lines[begin + 1 : end])
        try:
            blob = base64.b64decode(body, validate=True)
        except (binascii.Error, ValueError) as e:
            raise SSHSigFormatError(
                "invalid base64 in armored signature: {}".format(e)
            ) from e
        return cls.from_bytes(blob)

    @classmethod
    def load(cls, data):
        """
        Parse a signature in either armored or binary form (auto-detected).

        :param data: `str` or `bytes`; anything starting with the armor
            header (after leading whitespace) is treated as armored text,
            everything else as a binary blob.
        :raises SSHSigFormatError: on any parse failure.
        """
        probe = data.lstrip() if isinstance(data, (bytes, bytearray)) else None
        if isinstance(data, str):
            if data.lstrip().startswith(ARMORED_BEGIN):
                return cls.from_armored(data)
            raise SSHSigFormatError(
                "text input does not look like an armored SSH signature"
            )
        if probe is not None and probe.startswith(ARMORED_BEGIN.encode()):
            return cls.from_armored(data)
        return cls.from_bytes(data)

    def to_bytes(self):
        """
        Serialize to the binary sshsig blob form.
        """
        m = Message()
        m.add_string(self.public_key.asbytes())
        m.add_string(self.namespace.encode("utf-8"))
        m.add_string(self.reserved)
        m.add_string(self.hash_algorithm.encode("ascii"))
        m.add_string(self.signature)
        return (
            MAGIC_PREAMBLE
            + struct.pack(">I", SIG_VERSION)
            + m.asbytes()
        )

    def to_armored(self):
        """
        Serialize to the ASCII-armored form (compatible with OpenSSH).
        """
        b64 = base64.b64encode(self.to_bytes()).decode("ascii")
        wrapped = "\n".join(
            b64[i : i + _ARMOR_LINE_WIDTH]
            for i in range(0, len(b64), _ARMOR_LINE_WIDTH)
        )
        return "{}\n{}\n{}\n".format(ARMORED_BEGIN, wrapped, ARMORED_END)

    def signed_data(self, message):
        """
        Compute the exact byte string that is signed for ``message``.

        This is ``SSHSIG || string(namespace) || string(reserved) ||
        string(hash_algorithm) || string(H(message))``.
        """
        digest = HASH_ALGORITHMS[self.hash_algorithm](message).digest()
        m = Message()
        m.add_string(self.namespace.encode("utf-8"))
        m.add_string(self.reserved)
        m.add_string(self.hash_algorithm.encode("ascii"))
        m.add_string(digest)
        return MAGIC_PREAMBLE + m.asbytes()

    def verify(self, message, key=None, namespace=None):
        """
        Verify this signature over ``message``.

        :param bytes message: the data that was signed.
        :param PKey key: optional expected public key; when given, the
            signature's embedded key must match it exactly.
        :param str namespace: optional expected namespace; when given, the
            signature's namespace must match it.
        :returns: ``True`` on success.
        :raises SSHSigNamespaceError: if ``namespace`` does not match.
        :raises SSHSigKeyMismatchError: if ``key`` does not match the
            embedded public key.
        :raises SSHSigVerificationError: if the cryptographic signature is
            invalid for ``message``.
        """
        if namespace is not None and namespace != self.namespace:
            raise SSHSigNamespaceError(
                "namespace mismatch: expected {!r}, signature has {!r}".format(
                    namespace, self.namespace
                )
            )
        if key is not None and key.asbytes() != self.public_key.asbytes():
            raise SSHSigKeyMismatchError(
                "signature was made with a different public key"
            )
        # Normalize to a public-only key reconstructed from the embedded
        # blob: private-key-backed PKey instances are not guaranteed to be
        # verification-capable (e.g. Ed25519Key loaded from a private key
        # file only holds a signing key).
        verifier = _parse_public_key(self.public_key.asbytes())
        if not verifier.verify_ssh_sig(
            self.signed_data(message), Message(self.signature)
        ):
            raise SSHSigVerificationError(
                "signature did not verify against the embedded public key"
            )
        return True

    def __repr__(self):
        return "<SSHSignature {} ns={!r} key={}>".format(
            self.hash_algorithm, self.namespace, self.public_key.get_name()
        )


def sign_data(
    key, data, namespace=DEFAULT_NAMESPACE, hash_algorithm="sha512",
    sign_algorithm=None,
):
    """
    Sign ``data`` with ``key``, producing an OpenSSH-compatible sshsig
    signature.

    :param PKey key: private key to sign with (must satisfy
        ``key.can_sign()``). RSA keys produce ``rsa-sha2-512`` (or
        ``rsa-sha2-256``) signatures; Ed25519 keys produce ``ssh-ed25519``
        signatures.
    :param bytes data: the message to sign.
    :param str namespace: signature namespace; defaults to ``"file"`` like
        OpenSSH.
    :param str hash_algorithm: ``"sha512"`` (default) or ``"sha256"``.
    :param str sign_algorithm: optional signature algorithm override passed
        through to `.PKey.sign_ssh_data` (e.g. ``"rsa-sha2-256"``).
    :returns: an `SSHSignature`; use `.SSHSignature.to_armored` or
        `.SSHSignature.to_bytes` to serialize.
    :raises SSHSigFormatError: on an unsupported ``hash_algorithm``.
    :raises SSHException: if ``key`` cannot sign.
    """
    if hash_algorithm not in HASH_ALGORITHMS:
        raise SSHSigFormatError(
            "unsupported hash algorithm: {!r}".format(hash_algorithm)
        )
    if not key.can_sign():
        raise SSHException("key cannot sign (no private key material)")
    sig = SSHSignature(
        public_key=key,
        namespace=namespace,
        hash_algorithm=hash_algorithm,
        signature=b"",
    )
    inner = key.sign_ssh_data(
        sig.signed_data(data), algorithm=sign_algorithm
    )
    sig.signature = inner.asbytes()
    return sig


def verify_signature(
    data, signature, trusted_keys=None, namespace=DEFAULT_NAMESPACE
):
    """
    Verify an sshsig signature without raising; returns a structured result.

    :param bytes data: the signed message.
    :param signature: the signature, as armored text (`str` or `bytes`) or a
        binary blob (`bytes`), or an already-parsed `SSHSignature`.
    :param trusted_keys: optional iterable of `.PKey` objects (or public key
        blobs as `bytes`) the signer key must match; when given and the
        embedded key matches none of them, the result status is
        ``UNTRUSTED_KEY``. When ``None``, the embedded key is used as-is
        (i.e. only cryptographic validity is checked).
    :param str namespace: expected namespace; ``None`` skips the check.
    :returns: a `VerificationResult`.
    """
    try:
        if isinstance(signature, SSHSignature):
            sig = signature
        else:
            sig = SSHSignature.load(signature)
    except SSHSigFormatError as e:
        return VerificationResult(
            VerificationStatus.UNPARSEABLE, error=e
        )
    if trusted_keys is not None:
        trusted_blobs = {
            k.asbytes() if isinstance(k, PKey) else bytes(k)
            for k in trusted_keys
        }
        if sig.public_key.asbytes() not in trusted_blobs:
            return VerificationResult(
                VerificationStatus.UNTRUSTED_KEY,
                signature=sig,
                error=SSHSigKeyMismatchError(
                    "signature key is not in the trusted key set"
                ),
            )
    try:
        sig.verify(data, namespace=namespace)
    except SSHSigNamespaceError as e:
        return VerificationResult(
            VerificationStatus.NAMESPACE_MISMATCH, signature=sig, error=e
        )
    except SSHSigVerificationError as e:
        return VerificationResult(
            VerificationStatus.BAD_SIGNATURE, signature=sig, error=e
        )
    return VerificationResult(VerificationStatus.VALID, signature=sig)


def _parse_public_key(key_blob):
    """
    Build a `.PKey` from an SSH public key blob.

    :raises SSHSigFormatError: if the blob is not a usable public key.
    """
    try:
        key_type = Message(key_blob).get_text()
    except SSHException as e:
        raise SSHSigFormatError(
            "malformed public key in signature: {}".format(e)
        ) from e
    try:
        return PKey.from_type_string(key_type, key_blob)
    except Exception as e:
        raise SSHSigFormatError(
            "unusable public key of type {!r} in signature: {}".format(
                key_type, e
            )
        ) from e

//! Utility functions for computing bundle commitments

use blake2b_simd::{Hash as Blake2bHash, Params, State};

use crate::bundle::{Authorization, Authorized, Bundle, BundleFormat, BundleProtocol};

const ZCASH_ORCHARD_HASH_PERSONALIZATION: &[u8; 16] = b"ZTxIdOrchardHash";
const ZCASH_ORCHARD_ACTIONS_COMPACT_HASH_PERSONALIZATION: &[u8; 16] = b"ZTxIdOrcActCHash";
const ZCASH_ORCHARD_ACTIONS_MEMOS_HASH_PERSONALIZATION: &[u8; 16] = b"ZTxIdOrcActMHash";
const ZCASH_ORCHARD_ACTIONS_NONCOMPACT_HASH_PERSONALIZATION: &[u8; 16] = b"ZTxIdOrcActNHash";
const ZCASH_ORCHARD_SIGS_HASH_PERSONALIZATION: &[u8; 16] = b"ZTxAuthOrchaHash";
const ZCASH_IRONWOOD_HASH_PERSONALIZATION: &[u8; 16] = b"ZTxIdIronwd_Hash";
const ZCASH_IRONWOOD_ACTIONS_COMPACT_HASH_PERSONALIZATION: &[u8; 16] = b"ZTxIdIrnActCHash";
const ZCASH_IRONWOOD_ACTIONS_MEMOS_HASH_PERSONALIZATION: &[u8; 16] = b"ZTxIdIrnActMHash";
const ZCASH_IRONWOOD_ACTIONS_NONCOMPACT_HASH_PERSONALIZATION: &[u8; 16] = b"ZTxIdIrnActNHash";
const ZCASH_IRONWOOD_SIGS_HASH_PERSONALIZATION: &[u8; 16] = b"ZTxAuthIrnwdHash";

/// Whether a bundle commitment domain includes the anchor.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AnchorCommitment {
    /// Include the bundle anchor in the commitment.
    Include,
    /// Omit the bundle anchor from the commitment.
    Omit,
}

#[derive(Clone, Copy, Debug)]
struct BundleCommitmentPersonalizations {
    bundle: &'static [u8; 16],
    actions_compact: &'static [u8; 16],
    actions_memos: &'static [u8; 16],
    actions_noncompact: &'static [u8; 16],
    auth: &'static [u8; 16],
}

const ORCHARD_PERSONALIZATIONS: BundleCommitmentPersonalizations =
    BundleCommitmentPersonalizations {
        bundle: ZCASH_ORCHARD_HASH_PERSONALIZATION,
        actions_compact: ZCASH_ORCHARD_ACTIONS_COMPACT_HASH_PERSONALIZATION,
        actions_memos: ZCASH_ORCHARD_ACTIONS_MEMOS_HASH_PERSONALIZATION,
        actions_noncompact: ZCASH_ORCHARD_ACTIONS_NONCOMPACT_HASH_PERSONALIZATION,
        auth: ZCASH_ORCHARD_SIGS_HASH_PERSONALIZATION,
    };

const IRONWOOD_PERSONALIZATIONS: BundleCommitmentPersonalizations =
    BundleCommitmentPersonalizations {
        bundle: ZCASH_IRONWOOD_HASH_PERSONALIZATION,
        actions_compact: ZCASH_IRONWOOD_ACTIONS_COMPACT_HASH_PERSONALIZATION,
        actions_memos: ZCASH_IRONWOOD_ACTIONS_MEMOS_HASH_PERSONALIZATION,
        actions_noncompact: ZCASH_IRONWOOD_ACTIONS_NONCOMPACT_HASH_PERSONALIZATION,
        auth: ZCASH_IRONWOOD_SIGS_HASH_PERSONALIZATION,
    };

/// Parameters for computing Orchard-shaped bundle commitments.
///
/// This type selects the bundle-specific pieces of the commitment algorithm:
/// protocol personalization strings, flag-byte encoding format, and whether the
/// anchor is included in the effects or authorizing commitment. Transaction
/// versions decide how these bundle commitments are composed into the overall
/// transaction identifier.
#[derive(Clone, Copy, Debug)]
pub struct BundleCommitmentDomain {
    personalizations: BundleCommitmentPersonalizations,
    format: BundleFormat,
    effects_anchor: AnchorCommitment,
    auth_anchor: AnchorCommitment,
}

impl BundleCommitmentDomain {
    /// Constructs the Orchard commitment domain for the provided transaction format.
    pub const fn orchard(
        format: BundleFormat,
        effects_anchor: AnchorCommitment,
        auth_anchor: AnchorCommitment,
    ) -> Self {
        Self {
            personalizations: ORCHARD_PERSONALIZATIONS,
            format,
            effects_anchor,
            auth_anchor,
        }
    }

    /// Constructs the Ironwood commitment domain.
    ///
    /// Ironwood bundles always use the NU6.3 flag-byte format.
    pub const fn ironwood(effects_anchor: AnchorCommitment, auth_anchor: AnchorCommitment) -> Self {
        Self {
            personalizations: IRONWOOD_PERSONALIZATIONS,
            format: BundleFormat::Nu6_3,
            effects_anchor,
            auth_anchor,
        }
    }

    /// Constructs a commitment domain from a [`BundleProtocol`].
    ///
    /// [`BundleProtocol::LegacyOrchard`] and [`BundleProtocol::Orchard`] use
    /// Orchard personalization strings. [`BundleProtocol::Ironwood`] uses
    /// Ironwood personalization strings.
    pub const fn from_protocol(
        protocol: BundleProtocol,
        effects_anchor: AnchorCommitment,
        auth_anchor: AnchorCommitment,
    ) -> Self {
        match protocol {
            BundleProtocol::LegacyOrchard | BundleProtocol::Orchard => {
                Self::orchard(protocol.bundle_format(), effects_anchor, auth_anchor)
            }
            BundleProtocol::Ironwood => Self::ironwood(effects_anchor, auth_anchor),
        }
    }
}

fn hasher(personal: &[u8; 16]) -> State {
    Params::new().hash_length(32).personal(personal).to_state()
}

/// Write disjoint parts of each Orchard shielded action as 3 separate hashes
/// as defined in [ZIP-244: Transaction Identifier Non-Malleability][zip244]:
/// * \[(nullifier, cmx, ephemeral_key, enc_ciphertext\[..52\])*\] personalized
///   with ZCASH_ORCHARD_ACTIONS_COMPACT_HASH_PERSONALIZATION
/// * \[enc_ciphertext\[52..564\]*\] (memo ciphertexts) personalized
///   with ZCASH_ORCHARD_ACTIONS_MEMOS_HASH_PERSONALIZATION
/// * \[(cv, rk, enc_ciphertext\[564..\], out_ciphertext)*\] personalized
///   with ZCASH_ORCHARD_ACTIONS_NONCOMPACT_HASH_PERSONALIZATION
///
/// Then, hash these together along with (flags, value_balance_orchard, anchor_orchard),
/// personalized with ZCASH_ORCHARD_ACTIONS_HASH_PERSONALIZATION
///
/// [zip244]: https://zips.z.cash/zip-0244
pub(crate) fn hash_bundle_txid_data<A: Authorization, V: Copy + Into<i64>>(
    bundle: &Bundle<A, V>,
    format: BundleFormat,
) -> Blake2bHash {
    hash_bundle_txid_data_with_domain(
        bundle,
        BundleCommitmentDomain::orchard(format, AnchorCommitment::Include, AnchorCommitment::Omit),
    )
}

/// Construct the commitment to the effects of the specified bundle under the
/// given bundle commitment domain.
///
/// # Panics
///
/// Panics if the bundle flags cannot be encoded in the domain's bundle format.
pub(crate) fn hash_bundle_txid_data_with_domain<A: Authorization, V: Copy + Into<i64>>(
    bundle: &Bundle<A, V>,
    domain: BundleCommitmentDomain,
) -> Blake2bHash {
    let mut h = hasher(domain.personalizations.bundle);
    let mut ch = hasher(domain.personalizations.actions_compact);
    let mut mh = hasher(domain.personalizations.actions_memos);
    let mut nh = hasher(domain.personalizations.actions_noncompact);

    for action in bundle.actions().iter() {
        ch.update(&action.nullifier().to_bytes());
        ch.update(&action.cmx().to_bytes());
        ch.update(&action.encrypted_note().epk_bytes);
        ch.update(&action.encrypted_note().enc_ciphertext[..52]);

        mh.update(&action.encrypted_note().enc_ciphertext[52..564]);

        nh.update(&action.cv_net().to_bytes());
        nh.update(&<[u8; 32]>::from(action.rk()));
        nh.update(&action.encrypted_note().enc_ciphertext[564..]);
        nh.update(&action.encrypted_note().out_ciphertext);
    }

    h.update(ch.finalize().as_bytes());
    h.update(mh.finalize().as_bytes());
    h.update(nh.finalize().as_bytes());
    h.update(&[bundle.flags().to_byte(domain.format).expect(
        "cross-address-restricted bundles are not representable in pre-NU6.3 transaction formats",
    )]);
    h.update(&(*bundle.value_balance()).into().to_le_bytes());
    if domain.effects_anchor == AnchorCommitment::Include {
        h.update(&bundle.anchor().to_bytes());
    }
    h.finalize()
}

/// Construct the commitment for the absent bundle as defined in
/// [ZIP-244: Transaction Identifier Non-Malleability][zip244]
///
/// [zip244]: https://zips.z.cash/zip-0244
pub fn hash_bundle_txid_empty() -> Blake2bHash {
    hash_bundle_txid_empty_with_domain(BundleCommitmentDomain::orchard(
        BundleFormat::PreNu6_3,
        AnchorCommitment::Include,
        AnchorCommitment::Omit,
    ))
}

/// Construct the commitment for the absent bundle under the given bundle
/// commitment domain.
pub fn hash_bundle_txid_empty_with_domain(domain: BundleCommitmentDomain) -> Blake2bHash {
    hasher(domain.personalizations.bundle).finalize()
}

/// Construct the commitment to the authorizing data of an
/// authorized bundle as defined in [ZIP-244: Transaction
/// Identifier Non-Malleability][zip244]
///
/// [zip244]: https://zips.z.cash/zip-0244
pub(crate) fn hash_bundle_auth_data<V>(bundle: &Bundle<Authorized, V>) -> Blake2bHash {
    hash_bundle_auth_data_with_domain(
        bundle,
        BundleCommitmentDomain::orchard(
            BundleFormat::PreNu6_3,
            AnchorCommitment::Include,
            AnchorCommitment::Omit,
        ),
    )
}

/// Construct the commitment to the authorizing data of an authorized bundle
/// under the given bundle commitment domain.
pub(crate) fn hash_bundle_auth_data_with_domain<V>(
    bundle: &Bundle<Authorized, V>,
    domain: BundleCommitmentDomain,
) -> Blake2bHash {
    let mut h = hasher(domain.personalizations.auth);
    if domain.auth_anchor == AnchorCommitment::Include {
        h.update(&bundle.anchor().to_bytes());
    }
    h.update(bundle.authorization().proof().as_ref());
    for action in bundle.actions().iter() {
        h.update(&<[u8; 64]>::from(action.authorization()));
    }
    h.update(&<[u8; 64]>::from(
        bundle.authorization().binding_signature(),
    ));
    h.finalize()
}

/// Construct the commitment for an absent bundle as defined in
/// [ZIP-244: Transaction Identifier Non-Malleability][zip244]
///
/// [zip244]: https://zips.z.cash/zip-0244
pub fn hash_bundle_auth_empty() -> Blake2bHash {
    hash_bundle_auth_empty_with_domain(BundleCommitmentDomain::orchard(
        BundleFormat::PreNu6_3,
        AnchorCommitment::Include,
        AnchorCommitment::Omit,
    ))
}

/// Construct the commitment for absent authorizing data under the given bundle
/// commitment domain.
pub fn hash_bundle_auth_empty_with_domain(domain: BundleCommitmentDomain) -> Blake2bHash {
    hasher(domain.personalizations.auth).finalize()
}

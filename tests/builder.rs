#![cfg(feature = "circuit")]

use incrementalmerkletree::{Hashable, Marking, Retention};
use orchard::{
    builder::Builder,
    bundle::{Authorized, BatchValidator},
    circuit::{OrchardCircuitVersion, ProvingKey, VerifyingKey},
    keys::{FullViewingKey, PreparedIncomingViewingKey, Scope, SpendAuthorizingKey, SpendingKey},
    note::{ExtractedNoteCommitment, NoteVersion},
    note_encryption::OrchardDomain,
    tree::{MerkleHashOrchard, MerklePath},
    value::NoteValue,
    Address, Bundle, BundleProtocol,
};
use rand::rngs::OsRng;
use shardtree::{store::memory::MemoryShardStore, ShardTree};
use zcash_note_encryption::try_note_decryption;

/// Builds a single-leaf note commitment tree containing `cmx`, returning the tree
/// root and a witness for the leaf.
fn single_leaf_witness(cmx: &ExtractedNoteCommitment) -> (MerkleHashOrchard, MerklePath) {
    let leaf = MerkleHashOrchard::from_cmx(cmx);
    let mut tree: ShardTree<MemoryShardStore<MerkleHashOrchard, u32>, 32, 16> =
        ShardTree::new(MemoryShardStore::empty(), 100);
    tree.append(
        leaf,
        Retention::Checkpoint {
            id: 0,
            marking: Marking::Marked,
        },
    )
    .unwrap();
    let root = tree.root_at_checkpoint_id(&0).unwrap().unwrap();
    let position = tree.max_leaf_position(None).unwrap().unwrap();
    let merkle_path = tree
        .witness_at_checkpoint_id(position, &0)
        .unwrap()
        .unwrap();
    assert_eq!(root, merkle_path.root(leaf));
    (root, merkle_path.into())
}

fn verify_bundle(bundle: &Bundle<Authorized, i64>, vk: &VerifyingKey) {
    assert!(matches!(bundle.verify_proof(vk), Ok(())));
    let sighash: [u8; 32] = bundle.commitment().into();
    let bvk = bundle.binding_validating_key();
    for action in bundle.actions() {
        assert_eq!(action.rk().verify(&sighash, action.authorization()), Ok(()));
    }
    assert_eq!(
        bvk.verify(&sighash, bundle.authorization().binding_signature()),
        Ok(())
    );
}

/// Creates a builder for the given protocol over the empty-tree anchor, with a
/// single 5000-zat output to `recipient`.
fn output_only_builder(protocol: BundleProtocol, recipient: Address) -> Builder {
    let anchor = MerkleHashOrchard::empty_root(32.into()).into();
    let mut builder = Builder::new(protocol, anchor);
    assert_eq!(
        builder.add_output(None, recipient, NoteValue::from_raw(5000), [0u8; 512]),
        Ok(())
    );
    builder
}

#[test]
fn bundle_chain() {
    let mut rng = OsRng;
    let pk = ProvingKey::build(OrchardCircuitVersion::Ironwood);
    let vk = VerifyingKey::build(OrchardCircuitVersion::Ironwood);

    let sk = SpendingKey::from_bytes([0; 32]).unwrap();
    let fvk = FullViewingKey::from(&sk);
    let recipient = fvk.address_at(0u32, Scope::External);

    // Create a shielding bundle.
    let shielding_bundle: Bundle<_, i64> = {
        let builder = output_only_builder(BundleProtocol::Ironwood, recipient);
        let (unauthorized, bundle_meta) = builder.build(&mut rng).unwrap().unwrap();

        assert_eq!(
            unauthorized
                .decrypt_output_with_key(
                    bundle_meta
                        .output_action_index(0)
                        .expect("Output 0 can be found"),
                    &fvk.to_ivk(Scope::External)
                )
                .map(|(note, _, _)| note.value()),
            Some(NoteValue::from_raw(5000))
        );

        let sighash = unauthorized.commitment().into();
        let proven = unauthorized.create_proof(&pk, &mut rng).unwrap();
        proven.apply_signatures(rng, sighash, &[]).unwrap()
    };

    // Verify the shielding bundle.
    verify_bundle(&shielding_bundle, &vk);

    // Create a shielded bundle spending the previous output.
    let shielded_bundle: Bundle<_, i64> = {
        let ivk = PreparedIncomingViewingKey::new(&fvk.to_ivk(Scope::External));
        let (note, _, _) = shielding_bundle
            .actions()
            .iter()
            .find_map(|action| {
                let domain = OrchardDomain::for_action(action);
                try_note_decryption(&domain, &ivk, action)
            })
            .unwrap();

        // Use the tree with a single leaf.
        let cmx: ExtractedNoteCommitment = note.commitment().into();
        let (root, merkle_path) = single_leaf_witness(&cmx);

        let mut builder = Builder::new(BundleProtocol::Ironwood, root.into());
        assert_eq!(builder.add_spend(fvk, note, merkle_path), Ok(()));
        assert_eq!(
            builder.add_output(None, recipient, NoteValue::from_raw(5000), [0u8; 512]),
            Ok(())
        );
        let (unauthorized, _) = builder.build(&mut rng).unwrap().unwrap();
        let sighash = unauthorized.commitment().into();
        let proven = unauthorized.create_proof(&pk, &mut rng).unwrap();
        proven
            .apply_signatures(rng, sighash, &[SpendAuthorizingKey::from(&sk)])
            .unwrap()
    };

    // Verify the shielded bundle.
    verify_bundle(&shielded_bundle, &vk);
}

#[test]
fn builder_builds_for_ironwood_circuit_version() {
    let mut rng = OsRng;
    let ironwood_pk = ProvingKey::build(OrchardCircuitVersion::Ironwood);
    let ironwood_vk = VerifyingKey::build(OrchardCircuitVersion::Ironwood);

    let sk = SpendingKey::from_bytes([0; 32]).unwrap();
    let fvk = FullViewingKey::from(&sk);
    let recipient = fvk.address_at(0u32, Scope::External);

    let builder = output_only_builder(BundleProtocol::Ironwood, recipient);

    let (unauthorized, _) = builder.build::<i64>(&mut rng).unwrap().unwrap();
    assert_eq!(
        unauthorized.circuit_version(),
        OrchardCircuitVersion::Ironwood
    );

    let sighash: [u8; 32] = unauthorized.commitment().into();
    let proven = unauthorized.create_proof(&ironwood_pk, &mut rng).unwrap();
    let bundle = proven.apply_signatures(rng, sighash, &[]).unwrap();

    verify_bundle(&bundle, &ironwood_vk);
}

#[test]
fn builder_builds_for_orchard_protocol() {
    let mut rng = OsRng;
    let ironwood_pk = ProvingKey::build(OrchardCircuitVersion::Ironwood);
    let ironwood_vk = VerifyingKey::build(OrchardCircuitVersion::Ironwood);
    let fixed_vk = VerifyingKey::build(OrchardCircuitVersion::FixedPostNu6_2);

    let sk = SpendingKey::from_bytes([0; 32]).unwrap();
    let fvk = FullViewingKey::from(&sk);
    let recipient = fvk.address_at(0u32, Scope::Internal);

    let anchor = MerkleHashOrchard::empty_root(32.into()).into();
    let mut builder = Builder::new(BundleProtocol::Orchard, anchor);
    assert_eq!(
        builder.add_change_output(
            fvk.clone(),
            Some(fvk.to_ovk(Scope::Internal)),
            recipient,
            NoteValue::from_raw(5000),
            [0u8; 512],
        ),
        Ok(())
    );

    let (unauthorized, bundle_meta) = builder.build::<i64>(&mut rng).unwrap().unwrap();
    assert_eq!(
        unauthorized.circuit_version(),
        OrchardCircuitVersion::Ironwood
    );
    assert!(unauthorized.flags().spends_enabled());
    assert!(unauthorized.flags().outputs_enabled());
    assert!(unauthorized.flags().cross_address_disabled());
    assert_eq!(
        unauthorized
            .decrypt_output_with_key(
                bundle_meta
                    .output_action_index(0)
                    .expect("Output 0 can be found"),
                &fvk.to_ivk(Scope::Internal),
            )
            .map(|(note, _, _)| (note.value(), note.version())),
        Some((NoteValue::from_raw(5000), NoteVersion::V2))
    );

    let sighash: [u8; 32] = unauthorized.commitment().into();
    let proven = unauthorized.create_proof(&ironwood_pk, &mut rng).unwrap();
    let bundle = proven
        .apply_signatures(rng, sighash, &[SpendAuthorizingKey::from(&sk)])
        .unwrap();

    verify_bundle(&bundle, &ironwood_vk);
    assert!(bundle.verify_proof(&fixed_vk).is_err());
}

// Orchard pool coinbase: a single output-only action, no padding, spends disabled,
// disableCrossAddress unset. Downstream consensus policy decides whether this
// bundle type is accepted at a given height.
#[test]
fn orchard_coinbase_builder_constructs_v2_output() {
    let mut rng = OsRng;
    let sk = SpendingKey::from_bytes([0; 32]).unwrap();
    let fvk = FullViewingKey::from(&sk);
    let recipient = fvk.address_at(0u32, Scope::External);

    let anchor = MerkleHashOrchard::empty_root(32.into()).into();
    let mut builder = Builder::new_coinbase(BundleProtocol::Orchard, anchor);
    assert_eq!(
        builder.add_output(None, recipient, NoteValue::from_raw(5000), [0u8; 512]),
        Ok(())
    );

    let (unauthorized, bundle_meta) = builder.build::<i64>(&mut rng).unwrap().unwrap();

    assert_eq!(unauthorized.actions().len(), 1);
    assert!(!unauthorized.flags().spends_enabled());
    assert!(!unauthorized.flags().cross_address_disabled());
    assert_eq!(
        unauthorized.circuit_version(),
        OrchardCircuitVersion::Ironwood
    );

    let output_action_index = bundle_meta.output_action_index(0).unwrap();
    let (note, _, _) = unauthorized
        .decrypt_output_with_key(output_action_index, &fvk.to_ivk(Scope::External))
        .unwrap();
    assert_eq!(note.version(), NoteVersion::V2);
}

// Ironwood pool coinbase: a single output-only action, no padding, spends disabled,
// disableCrossAddress unset. Verifies under the Ironwood VK; rejected by FixedPostNu6_2.
#[test]
fn ironwood_coinbase_proves_and_verifies() {
    let mut rng = OsRng;
    let ironwood_pk = ProvingKey::build(OrchardCircuitVersion::Ironwood);
    let ironwood_vk = VerifyingKey::build(OrchardCircuitVersion::Ironwood);
    let fixed_vk = VerifyingKey::build(OrchardCircuitVersion::FixedPostNu6_2);

    let sk = SpendingKey::from_bytes([0; 32]).unwrap();
    let fvk = FullViewingKey::from(&sk);
    let recipient = fvk.address_at(0u32, Scope::External);

    let anchor = MerkleHashOrchard::empty_root(32.into()).into();
    let mut builder = Builder::new_coinbase(BundleProtocol::Ironwood, anchor);
    assert_eq!(
        builder.add_output(None, recipient, NoteValue::from_raw(5000), [0u8; 512]),
        Ok(())
    );

    let (unauthorized, bundle_meta) = builder.build::<i64>(&mut rng).unwrap().unwrap();

    assert_eq!(unauthorized.actions().len(), 1);
    assert!(!unauthorized.flags().spends_enabled());
    assert!(!unauthorized.flags().cross_address_disabled());
    assert_eq!(
        unauthorized.circuit_version(),
        OrchardCircuitVersion::Ironwood
    );
    let output_action_index = bundle_meta.output_action_index(0).unwrap();
    let (note, _, _) = unauthorized
        .decrypt_output_with_key(output_action_index, &fvk.to_ivk(Scope::External))
        .unwrap();
    assert_eq!(note.version(), NoteVersion::V3);

    let sighash: [u8; 32] = unauthorized.commitment().into();
    let proven = unauthorized.create_proof(&ironwood_pk, &mut rng).unwrap();
    let bundle = proven.apply_signatures(rng, sighash, &[]).unwrap();

    verify_bundle(&bundle, &ironwood_vk);
    assert!(bundle.verify_proof(&fixed_vk).is_err());
}

// An Ironwood bundle chain: an ordinary shielding bundle, followed by a bundle
// that disables cross-address transfers, withdraws part of the shielded value,
// and retains the rest as wallet-controlled change.
#[test]
fn ironwood_restricted_bundle_chain() {
    let mut rng = OsRng;
    let ironwood_pk = ProvingKey::build(OrchardCircuitVersion::Ironwood);
    let ironwood_vk = VerifyingKey::build(OrchardCircuitVersion::Ironwood);
    let fixed_vk = VerifyingKey::build(OrchardCircuitVersion::FixedPostNu6_2);

    let sk = SpendingKey::from_bytes([0; 32]).unwrap();
    let fvk = FullViewingKey::from(&sk);
    let recipient = fvk.address_at(0u32, Scope::External);

    let shielding_bundle: Bundle<_, i64> = {
        let builder = output_only_builder(BundleProtocol::Ironwood, recipient);

        let (unauthorized, _) = builder.build(&mut rng).unwrap().unwrap();
        let sighash = unauthorized.commitment().into();
        let proven = unauthorized.create_proof(&ironwood_pk, &mut rng).unwrap();
        proven.apply_signatures(rng, sighash, &[]).unwrap()
    };

    verify_bundle(&shielding_bundle, &ironwood_vk);
    assert!(shielding_bundle.verify_proof(&fixed_vk).is_err());

    let change_addr = fvk.address_at(0u32, Scope::Internal);
    let restricted_bundle: Bundle<_, i64> = {
        let ivk = PreparedIncomingViewingKey::new(&fvk.to_ivk(Scope::External));
        let (note, _, _) = shielding_bundle
            .actions()
            .iter()
            .find_map(|action| {
                let domain = OrchardDomain::for_action(action);
                try_note_decryption(&domain, &ivk, action)
            })
            .unwrap();

        let cmx: ExtractedNoteCommitment = note.commitment().into();
        let (root, merkle_path) = single_leaf_witness(&cmx);

        let mut builder = Builder::new(BundleProtocol::Orchard, root.into());
        assert_eq!(builder.add_spend(fvk.clone(), note, merkle_path), Ok(()));
        assert_eq!(
            builder.add_change_output(
                fvk.clone(),
                Some(fvk.to_ovk(Scope::Internal)),
                change_addr,
                NoteValue::from_raw(3000),
                [0u8; 512],
            ),
            Ok(())
        );
        let (unauthorized, bundle_meta) = builder.build(&mut rng).unwrap().unwrap();

        assert_eq!(unauthorized.actions().len(), 2);
        assert_ne!(
            bundle_meta.spend_action_index(0),
            bundle_meta.output_action_index(0)
        );
        assert_eq!(
            unauthorized
                .decrypt_output_with_key(
                    bundle_meta
                        .output_action_index(0)
                        .expect("Output 0 can be found"),
                    &fvk.to_ivk(Scope::Internal),
                )
                .map(|(note, recipient, _)| (note.value(), recipient)),
            Some((NoteValue::from_raw(3000), change_addr))
        );

        let sighash = unauthorized.commitment().into();
        let proven = unauthorized.create_proof(&ironwood_pk, &mut rng).unwrap();
        proven
            .apply_signatures(rng, sighash, &[SpendAuthorizingKey::from(&sk)])
            .unwrap()
    };

    assert_eq!(restricted_bundle.value_balance(), &2000);
    verify_bundle(&restricted_bundle, &ironwood_vk);
    assert!(restricted_bundle.verify_proof(&fixed_vk).is_err());

    let mut validator = BatchValidator::new();
    validator.add_bundle(&restricted_bundle, restricted_bundle.commitment().into());
    assert!(validator.validate(&ironwood_vk, rng));

    let mut validator = BatchValidator::new();
    validator.add_bundle(&restricted_bundle, restricted_bundle.commitment().into());
    assert!(!validator.validate(&fixed_vk, rng));
}

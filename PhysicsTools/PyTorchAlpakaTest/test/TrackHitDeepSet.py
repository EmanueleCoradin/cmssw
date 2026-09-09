import torch
import torch.nn as nn


def build_hit_to_track(hit_offsets: torch.Tensor) -> torch.Tensor:
    """
    Construct the mapping once, outside the model.

    hit_offsets has shape [ntracks + 1].

    Hits belonging to track i are:
        [hit_offsets[i], hit_offsets[i + 1])

    Returns:
        hit_to_track: [nhits]
    """
    hit_counts = hit_offsets[1:] - hit_offsets[:-1]

    return torch.repeat_interleave(
        torch.arange(
            hit_counts.numel(),
            dtype=torch.int64,
            device=hit_offsets.device,
        ),
        hit_counts,
    )


class TrackHitDeepSet(nn.Module):
    def __init__(
        self,
        track_feature_dim: int,
        hit_feature_dim: int,
        embedding_dim: int,
    ):
        super().__init__()

        self.track_encoder = nn.Sequential(
            nn.Linear(track_feature_dim, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, embedding_dim),
            nn.ReLU(),
        )

        self.hit_encoder = nn.Sequential(
            nn.Linear(hit_feature_dim, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, embedding_dim),
            nn.ReLU(),
        )

        self.classifier = nn.Sequential(
            nn.Linear(2 * embedding_dim, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, 1),
        )

    def forward(
        self,
        track_features: torch.Tensor,
        hit_features: torch.Tensor,
        hit_to_track: torch.Tensor,
        track_begin: torch.Tensor,
    ) -> torch.Tensor:
        """
        Arguments
        ---------
        track_features:
            [batch_size, track_feature_dim]

            Contains only the tracks in the current mini-batch.

        hit_features:
            [nhits, hit_feature_dim]

            Contains all the event hits.

        hit_to_track:
            [nhits]

            Global track index associated with each hit. It is precomputed
            once before running the mini-batches.

        track_begin:
            Scalar int64 tensor containing the global index of the first
            track in the current batch.

        Returns
        -------
        scores:
            [batch_size, 1]
        """
        batch_size = track_features.size(0)
        track_end = track_begin + batch_size

        # Select the hits associated with tracks in this batch.
        #
        # This scans hit_to_track, but the expensive hit_encoder below is
        # evaluated only for the selected hits.
        hit_mask = torch.logical_and(
            hit_to_track >= track_begin,
            hit_to_track < track_end,
        )

        batch_hit_features = hit_features[hit_mask]

        # Convert global track indices into batch-local indices [0, B).
        batch_hit_to_track = hit_to_track[hit_mask] - track_begin

        # Only hits belonging to the current batch are embedded.
        hit_embeddings = self.hit_encoder(batch_hit_features)

        pooled_hits = torch.zeros(
            (batch_size, hit_embeddings.size(1)),
            dtype=hit_embeddings.dtype,
            device=hit_embeddings.device,
        )

        # Sum the embeddings belonging to each track.
        pooled_hits.index_add_(
            0,
            batch_hit_to_track,
            hit_embeddings,
        )

        track_embeddings = self.track_encoder(track_features)

        combined_embeddings = torch.cat(
            (track_embeddings, pooled_hits),
            dim=1,
        )

        logits = self.classifier(combined_embeddings)

        return torch.sigmoid(logits)


def main() -> int:
    torch.manual_seed(1234)

    ntracks = 8
    track_feature_dim = 3
    hit_feature_dim = 3

    # Number of hits associated with each track:
    hit_counts = torch.tensor(
        [2, 3, 1, 4, 1, 2, 3],
        dtype=torch.int64,
    )

    # hits of track i are:
    #     hit_offsets[i] ... hit_offsets[i + 1] - 1
    hit_offsets = torch.cat(
        (
            torch.zeros(1, dtype=torch.int64),
            torch.cumsum(hit_counts, dim=0),
        )
    )
    
    nhits = int(hit_offsets[-1])

    track_features = torch.randn(
        ntracks,
        track_feature_dim,
    )

    hit_features = torch.randn(
        nhits,
        hit_feature_dim,
    )

    # This operation is performed only once per event.
    hit_to_track = build_hit_to_track(hit_offsets)
    
    track_begin = torch.tensor([0])

    model = TrackHitDeepSet(
        track_feature_dim=track_feature_dim,
        hit_feature_dim=hit_feature_dim,
        embedding_dim=16,
    )

    model.eval()

    with torch.no_grad():
        model_input = (
            track_features,
            hit_features,
            hit_to_track,
            track_begin
        )
        y = model(*model_input)
        print("Input: ")
        for tensor in model_input:
            print(tensor)
        print("Output: ", y)

    tm = torch.jit.trace(model, model_input)
    tm.save("TrackHitDeepSet.pt")

if __name__ == "__main__":
    main()
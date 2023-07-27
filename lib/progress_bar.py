from tqdm import tqdm


class ProgressBar(tqdm):
    def __init__(
        self,
        episodes: int,
        batches: int,
        sequences_per_batch: int,
        prefix: str = None,
        leave: bool = None,
    ):
        self.prefix = prefix
        self.episodes = episodes
        self.batches = batches
        self.sequences_per_batch = sequences_per_batch
        self.total_iterations = episodes * batches * sequences_per_batch

        self.curr_ep = 1
        self.curr_b = 1
        self.curr_s = 1
        super().__init__(
            total=self.total_iterations,
            leave=leave,
            desc=self.gen_description(),
        )

    def gen_description(self):
        start = f"{self.prefix}|" if self.prefix else ""
        return f"{start}Episode {self.curr_ep}/{self.episodes}|Batch {self.curr_b}/{self.batches}|Sequence {self.curr_s}/{self.sequences_per_batch}"

    def update(self, ep: int = 0, batch: int = 0, seq: int = 1):
        self.curr_ep += ep
        self.curr_b += batch
        self.curr_s += seq

        if self.curr_s > self.sequences_per_batch:
            overflow_s = self.curr_s - 1
            self.curr_b += overflow_s // self.sequences_per_batch
            self.curr_s = overflow_s % self.sequences_per_batch

        if self.curr_b > self.batches:
            overflow_b = self.curr_b - 1
            self.curr_ep += overflow_b // self.batches
            self.curr_b = overflow_b % self.batches

        if self.curr_ep > self.episodes:
            print("Increment exceeds maximum.")

        # Update the super class with the calculated increments
        e = ep * self.batches * self.sequences_per_batch
        b = batch * self.sequences_per_batch
        super().update(e + b + seq)

        # Update the description
        super().set_description(self.gen_description())

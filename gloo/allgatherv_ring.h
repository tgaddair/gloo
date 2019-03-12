/**
 * Copyright (c) 2017-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <stddef.h>
#include <string.h>

#include "gloo/algorithm.h"
#include "gloo/context.h"

namespace gloo {

// AllgathervRing is similar to MPI_Allgatherv where all processes receive the
// buffers (inPtrs) of varying lengths (counts) from all other processes.
// The caller needs to pass a preallocated receive buffer (outPtr) of size equal
// to the context size x the total size of the send buffers (inPtrs) where the
// send buffers of the process with rank = k will be written to
// outPtr[k * number of input buffers * count] consecutively.
template <typename T>
class AllgathervRing : public Algorithm {
public:
  AllgathervRing(
      const std::shared_ptr<Context>& context,
      const std::vector<const T*>& inPtrs,
      T* outPtr,
      std::vector<int> counts)
      : Algorithm(context),
        inPtrs_(inPtrs),
        outPtr_(outPtr),
        counts_(std::move(counts)),
        leftPair_(this->getLeftPair()),
        rightPair_(this->getRightPair()) {
    bytes_.resize(counts_.size());
    displacements_.resize(counts_.size());
    int offset = 0;
    size_t totalBytes = 0;
    for (int i = 0; i < counts_.size(); i++) {
      bytes_[i] = counts_[i] * sizeof(T);
      displacements_[i] = offset;
      offset += counts_[i] * inPtrs.size();
      totalBytes += bytes_[i] * inPtrs_.size();
    }

    auto slot = this->context_->nextSlot();
    sendDataBuf_ = rightPair_->createSendBuffer(slot, outPtr_, totalBytes);
    recvDataBuf_ = leftPair_->createRecvBuffer(slot, outPtr_, totalBytes);

    auto notificationSlot = this->context_->nextSlot();
    sendNotificationBuf_ =
        leftPair_->createSendBuffer(notificationSlot, &dummy_, sizeof(dummy_));
    recvNotificationBuf_ =
        rightPair_->createRecvBuffer(notificationSlot, &dummy_, sizeof(dummy_));
  }

  virtual ~AllgathervRing() {}

  void run() {
    const int rank = this->contextRank_;
    const int numRounds = this->contextSize_ - 1;

    // Copy local buffers.
    for (int i = 0; i < inPtrs_.size(); i++) {
      memcpy(outPtr_ + displacements_[rank] + i * counts_[rank], inPtrs_[i], bytes_[rank]);
    }

    // We send input buffers in order.
    for (int i = 0; i < inPtrs_.size(); i++) {
      // We start every iteration by sending local buffer.
      int inRank = rank;
      for (int round = 0; round < numRounds; round++) {
        const int sendOffset = displacements_[inRank] + i * counts_[inRank];
        sendDataBuf_->send(
            sendOffset * sizeof(T), bytes_[inRank], sendOffset * sizeof(T));
        recvDataBuf_->waitRecv();

        // Nodes receive data from the left node in every round and forward it
        // to the right node.
        inRank = (numRounds - round + rank) % this->contextSize_;

        // Send notification to node on the left that this node is ready for an
        // inbox write.
        sendNotificationBuf_->send();

        // Wait for notification from node on the right.
        recvNotificationBuf_->waitRecv();
      }
    }
  }

private:
  const std::vector<const T*> inPtrs_;
  T* outPtr_;
  const std::vector<int> counts_;
  std::vector<size_t> bytes_;
  std::vector<int> displacements_;

  std::unique_ptr<transport::Pair>& leftPair_;
  std::unique_ptr<transport::Pair>& rightPair_;

  std::unique_ptr<transport::Buffer> sendDataBuf_;
  std::unique_ptr<transport::Buffer> recvDataBuf_;

  int dummy_;

  std::unique_ptr<transport::Buffer> sendNotificationBuf_;
  std::unique_ptr<transport::Buffer> recvNotificationBuf_;
};

}  // namespace gloo
